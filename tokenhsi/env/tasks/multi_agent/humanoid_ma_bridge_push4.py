from pathlib import Path

from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_angle_axis
import torch

from env.tasks.multi_agent.humanoid_ma_bridge import HumanoidMABridge
from env.tasks.multi_agent.humanoid_ma_bridge_push import HumanoidMABridgePush
from env.tasks.multi_agent.bridge_geometry import vertical_height
from env.tasks.multi_agent.bridge_push4_scene import task_layout
from utils.bridge_push4_spec import episode_graph, sample_roles
from utils.push_box_asset import create_plate_box
from utils.push_navigation import sample_box_properties


class HumanoidMABridgePush4(HumanoidMABridgePush):
    _extra_slots = 4

    def _sample_bridge_geometry(self, n):
        sizes, masses = sample_box_properties(4*n, self._push_options['box'], self.device)
        self._push_cube_size, self._push_cube_mass = sizes.view(n, 4, 3), masses.view(n, 4)
        side = self._push_cube_size[:, 0]
        return side[:, 0]+2*self._bridge_cfg['clearance'], side, side[:, 2]+self._bridge_cfg['top_drop']

    def _create_bridge_assets(self):
        n, p, slot = self.num_envs, self.num_physical_objects, self._bridge_slot
        self._box_size[:, slot:] = self._push_cube_size
        self._box_density.view(n, p)[:, slot:] = self._push_cube_mass/self._push_cube_size.prod(-1)
        directory = Path(self.cfg['args'].output_path)/'push_assets'
        for env in range(n):
            for cube in range(4):
                index = env*p+slot+cube
                options = gymapi.AssetOptions()
                options.angular_damping = options.linear_damping = .01
                options.max_angular_velocity = 100.
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                self._box_assets[index] = create_plate_box(
                    self.gym, self.sim, self._push_cube_size[env, cube].tolist(),
                    float(self._push_cube_mass[env, cube]), options,
                    self._push_options['box']['bottom_plate'], directory, index)

    def _build_platform_state_tensors(self):
        super()._build_platform_state_tensors()
        n, m = self.num_envs, self.num_agents
        actors = self.get_num_actors_per_env()
        self._near_stage_states = self._root_states.view(n, actors, -1)[:, self._near_stage_handle]
        self._near_stage_actor_ids = actors*torch.arange(n, device=self.device, dtype=torch.int32)+self._near_stage_handle
        self._roles = torch.zeros(n, m, dtype=torch.long, device=self.device)
        self._tasks4 = torch.ones_like(self._roles)
        self._task_surface = torch.zeros_like(self._roles)
        self._start_surface = torch.zeros_like(self._roles)
        self._push_goal = torch.zeros(n, m, 3, device=self.device)
        self._task_goal = torch.zeros_like(self._push_goal)
        self._independent_counts = torch.zeros(4, device=self.device)
        self._surface_counts = torch.zeros(3, device=self.device)
        self._bridge_near_start = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._bridge_aligned_start = torch.zeros_like(self._bridge_near_start)
        self._bridge_start_counts = torch.zeros(3, device=self.device)
        self._bridge_start_metrics = torch.zeros(3, 4, device=self.device)

    def _sample_episode_graph(self, ids):
        n, m, slot = len(ids), self.num_agents, self._bridge_slot
        roles, tasks, surface, start = sample_roles(n, self.device)
        mixture = self._bridge_cfg.get('start_mixture')
        if mixture is not None:
            self._bridge_near_start[ids] = torch.rand(n, device=self.device) < mixture['near_probability']
            aligned = mixture.get('near_aligned_probability', 0.)
            if aligned:
                self._bridge_aligned_start[ids] = self._bridge_near_start[ids] & (torch.rand(n, device=self.device) < aligned)
            mode = self._bridge_near_start[ids].long()+self._bridge_aligned_start[ids].long()
            self._bridge_start_counts += torch.bincount(mode, minlength=3)
        people = torch.arange(m, device=self.device)[None].expand(n, -1)
        primary = people+torch.where(tasks == 1, m, torch.where(tasks == 2, 2*m, 0))
        cube = slot+torch.arange(m, device=self.device)[None].expand(n, -1)
        cube = cube.scatter(1, roles, slot+torch.arange(m, device=self.device)[None].expand(n, -1))
        primary = torch.where(tasks == 4, cube, primary)
        used = torch.zeros(n, 4*m+4, dtype=torch.bool, device=self.device)
        used.scatter_(1, primary, True)
        noise = torch.rand(n, 4*m, device=self.device).masked_fill(used[:, :4*m], 2.)
        spare = noise.argsort(-1)[:, :4]
        self._logical_box_order[ids] = torch.cat((primary, spare), -1)
        self._agent_box_assignment[ids] = primary
        self._roles[ids], self._tasks4[ids] = roles, tasks
        self._task_surface[ids], self._start_surface[ids] = surface, start
        self._templates[ids] = tasks.clamp_max(3)
        self._crosser[ids] = people == roles[:, 1:2]
        self._pushing[ids] = tasks == 4
        self._floor_start[ids] = start == 0
        self._push_payload[ids] = primary
        self._bridge_scene[ids] = 0
        self._before_distance_modes[ids], self._before_near_pairs[ids] = 0, False
        self._task_token_count[ids] = m
        graph, self._before_predecessor[ids] = episode_graph(tasks, roles, self.num_objects)
        self._finish_episode_graph(ids, graph, tasks.clamp_max(3))
        independent = tasks.gather(1, roles[:, 2:]).flatten()-1
        self._independent_counts += torch.bincount(independent, minlength=4)
        self._surface_counts += torch.bincount(surface.gather(1, roles[:, 2:]).flatten(), minlength=3)

    def _reset_boxes(self, ids):
        self._reset_all_boxes_random_arena(ids)
        self._platform_pos[ids] = self._platform_default_pos[ids]
        n, m = len(ids), self.num_agents
        payload = self._agent_box_assignment[ids]
        size = self._box_size[ids[:, None], payload]
        layout = task_layout(self._roles[ids], self._tasks4[ids], self._task_surface[ids],
            self._start_surface[ids], size, self._stage_half_size[ids], self._far_stage_half_size[ids],
            self._bridge_size[ids], self._bridge_cfg,
            self._bridge_near_start[ids] if 'start_mixture' in self._bridge_cfg else None,
            self._bridge_aligned_start[ids] if 'start_mixture' in self._bridge_cfg else None)
        origin, height = self._env_origins[ids], self._stage_height[ids]
        for state, key in ((self._near_stage_states, 'near'), (self._far_stage_states, 'far')):
            state[ids] = 0.
            state[ids, :2] = origin[:, :2]+layout[key]
            state[ids, 2] = origin[:, 2]+height/2
            state[ids, 6] = 1.
        lift = (self._task_surface[ids] > 0)*height[:, None]
        box = torch.zeros(n, m, 13, device=self.device)
        box[..., :2] = origin[:, None, :2]+layout['boxes']
        box[..., 2] = origin[:, None, 2]+lift+size[..., 2]/2+.002
        axis = torch.tensor([0., 0., 1.], device=self.device).expand(n*m, -1)
        box[..., 3:7] = quat_from_angle_axis(layout['yaw'].flatten(), axis).view(n, m, 4)
        self._box_states[ids[:, None], payload] = box
        self._task_goal[ids, :, :2] = origin[:, None, :2]+layout['goals']
        self._task_goal[ids, :, 2] = origin[:, None, 2]+lift+size[..., 2]/2
        self._push_goal[ids] = self._task_goal[ids]
        self._bridge_start[ids] = self._bridge_states[ids, :3]
        self._bridge_goal[ids] = self._task_goal[ids, self._roles[ids, 0]]
        self._bridge_connected[ids], self._bridge_present[ids] = False, False
        self._bridge_crossed[ids] = False
        self._bridge_wait[ids], self._bridge_duration[ids] = 0., 1.
        xy = origin[:, None, :2]+layout['humans']
        target = box[..., :2]
        start_lift = (self._start_surface[ids] > 0)*height[:, None]
        self._place_humans(ids, xy, self._face(xy, target), start_lift)
        self._push_rsi[ids] = self._pushing[ids]
        self._push_hits[ids] = False
        self._start_push_rsi(ids)
        occupied = torch.cat((layout['boxes'], layout['goals']), 1)
        for column in range(m, self.num_objects):
            physical = self._logical_box_order[ids, column]
            radius = self._box_size[ids, physical, :2].norm(dim=-1)/2
            pending = torch.ones(n, device=self.device, dtype=torch.bool)
            for _ in range(512):
                point = layout['near'] + torch.stack((
                    -self._stage_half_size[ids, 0]-2-torch.rand(n, device=self.device)*5,
                    (torch.rand(n, device=self.device)*2-1)*8), -1)
                valid = ((point[:, None]-occupied).norm(dim=-1) > radius[:, None]+1.3).all(-1)
                valid &= ((point[:, None]-layout['humans']).norm(dim=-1) > radius[:, None]+.6).all(-1)
                keep = pending & valid
                self._box_states[ids[keep], physical[keep]] = 0.
                self._box_states[ids[keep], physical[keep], :2] = point[keep]+origin[keep, :2]
                self._box_states[ids[keep], physical[keep], 2] = (origin[:, 2]+
                    self._box_size[ids, physical, 2]/2+.002)[keep]
                self._box_states[ids[keep], physical[keep], 6] = 1.
                pending &= ~valid
                if not pending.any():
                    break
            if pending.any():
                raise RuntimeError('Bridge PUSH4 distractors exhausted free floor samples')
            occupied = torch.cat((occupied, (self._box_states[ids, physical, :2]-origin[:, :2])[:, None]), 1)

    def _start_push_rsi(self, ids):
        super()._start_push_rsi(ids)
        row, human = self._push_rsi[ids].nonzero(as_tuple=True)
        env = ids[row]
        lift = (self._start_surface[env, human] > 0)*self._stage_height[env]+self._env_origins[env, 2]
        self._humanoid_root_states[env, human, 2] += lift
        self._kinematic_humanoid_rigid_body_states[env, human, :, 2] += lift[:, None]

    def _reset_task(self, ids):
        super(HumanoidMABridge, self)._reset_task(ids)
        self._tar_pos[ids] = self._task_goal[ids]

    def _extra_reset_actor_ids(self, ids):
        return [self._near_stage_actor_ids[ids], self._far_stage_actor_ids[ids]]

    def _terrain_height(self, xy, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        height = self._stage_height[ids, None]/2
        a = torch.cat((self._stage_half_size[ids], height), -1)
        b = torch.cat((self._far_stage_half_size[ids], height), -1)
        states = torch.stack((self._near_stage_states[ids], self._far_stage_states[ids], self._bridge_states[ids]), 1)
        half = torch.stack((a, b, self._bridge_size[ids]/2), 1)
        return vertical_height(xy, states[..., :3], states[..., 3:7], half, self._env_origins[ids, 2])

    def _record_bridge_metrics(self, fallen):
        root = self._rigid_body_pos[..., 0, :]
        near = root[..., :2]-self._near_stage_states[:, None, :2]
        on_near = (near.abs() <= self._stage_half_size[:, None]).all(-1)
        near_gap = on_near & (self._stage_half_size[:, None, 0]-near[..., 0] < 1.)
        far_xy = root[..., :2]-self._far_stage_states[:, None, :2]
        far = (far_xy.abs() <= self._far_stage_half_size[:, None]).all(-1)
        far &= root[..., 2] > self._stage_height[:, None]+self._env_origins[:, None, 2]+.6
        stopped = self._rigid_body_vel[..., 0, :2].norm(dim=-1) < .15
        arrived, crosser = self._bridge_arrived(), self._crosser
        waiting = ~arrived[:, None] & crosser
        self._bridge_crossed |= far & crosser
        ending = self.reset_buf.bool()
        self._bridge_metrics += torch.stack((crosser.sum(), waiting.sum(),
            (waiting & near_gap & stopped).sum(), (fallen & waiting).sum(),
            (fallen & arrived[:, None] & crosser).sum(), (far & crosser).sum(), ending.sum(),
            (ending & arrived).sum(), (self._bridge_crossed & ending[:, None]).sum(),
            ending.sum(), ending.sum(), (ending & arrived).sum())).float()
        if 'start_mixture' in self._bridge_cfg:
            modes = self._bridge_near_start.long()+self._bridge_aligned_start.long()
            for mode in range(3):
                selected = ending & (modes == mode)
                self._bridge_start_metrics[mode] += torch.stack((selected.sum(),
                    (selected & arrived).sum(), (selected & self._bridge_crossed.any(-1)).sum(),
                    (selected & (fallen & crosser).any(-1)).sum())).float()

    def _consume_sampling_diagnostics(self):
        result = HumanoidMABridge._consume_sampling_diagnostics(self)
        if 'start_mixture' in self._bridge_cfg:
            for mode, name in enumerate(('original', 'near_random', 'near_aligned')):
                prefix = 'bridge/start_'+name
                result[prefix+'_fraction'] = self._bridge_start_counts[mode]/self._bridge_start_counts.sum().clamp_min(1)
                episodes = self._bridge_start_metrics[mode, 0].clone()
                result[prefix+'_episodes'] = episodes
                for column, metric in enumerate(('arrived', 'crossed', 'crosser_falls'), 1):
                    result[prefix+'_'+metric] = self._bridge_start_metrics[mode, column]/episodes.clamp_min(1)
            result['bridge/start_near_fraction'] = self._bridge_start_counts[1:].sum()/self._bridge_start_counts.sum().clamp_min(1)
            self._bridge_start_counts.zero_()
            self._bridge_start_metrics.zero_()
        for values, names, prefix in ((self._independent_counts, ('sit', 'climb', 'carry', 'push'), 'independent'),
                                      (self._surface_counts, ('floor', 'stage_a', 'stage_b'), 'surface')):
            for index, name in enumerate(names):
                result['bridge/'+prefix+'_'+name+'_fraction'] = values[index]/values.sum().clamp_min(1)
            values.zero_()
        return result
