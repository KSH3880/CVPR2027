import copy
import os
import numpy as np
from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_bridge import HumanoidMABridge, HIDDEN, LIFT
from env.tasks.multi_agent.room_geometry import room_walls, sample_partition_layout, compute_wall_cpa_collision_penalty
from env.tasks.multi_agent.mixed_sensor_geometry import sample_stage_goals, load_foot_vertices, lowest_foot_height
from utils.edge_scenario_spec import agent_object_indices, agent_goal_indices, classify_templates

SCENARIOS = ('cliff', 'bridge', 'room')
TASKS = ('holding', 'sit', 'climb', 'holding_at')


class HumanoidMAMixedSensor(HumanoidMABridge):
    _supports_holding_at = True
    _fixed_far_stage = True

    def __init__(self, cfg, *args, **kwargs):
        self._room = cfg['env']['room']
        self._mixture = cfg['env']['scenarioMixture']
        probabilities = [self._mixture[name] for name in SCENARIOS]
        if min(probabilities) < 0 or not np.isclose(sum(probabilities), 1.):
            raise ValueError('Scenario probabilities must be nonnegative and sum to one')
        # Evaluation-only scene choice; sensor metadata keeps the trained mixture so checkpoints still load.
        scenario = os.environ.get('SCENARIO')
        if scenario:
            if not cfg['args'].eval or scenario not in SCENARIOS:
                raise ValueError('SCENARIO selects cliff, bridge or room for evaluation only')
            probabilities = [float(name == scenario) for name in SCENARIOS]
        self._wall_collision_ttc_discount = cfg['env']['wallCollisionTTCDiscount']
        if not 0 < self._wall_collision_ttc_discount <= 1:
            raise ValueError('wallCollisionTTCDiscount must be in (0, 1]')
        super().__init__(cfg, *args, **kwargs)
        self._arena_scale = cfg['env']['observationArenaScale']
        self.sensor_metadata.update(version=8 if self._square_sensor else 7, room=copy.deepcopy(self._room),
                                    scenario_mixture=copy.deepcopy(self._mixture),
                                    holding_at_goals='stage_a_or_room_opposite_side')
        self._scenario_probabilities = torch.tensor(probabilities, device=self.device)
        self._scenario_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._floor_height = torch.full((self.num_envs,), self._cliff['height'], device=self.device)
        self._scene_origins = self._env_origins[:, None] + self._scene_offsets[None]
        roots = self._root_states.view(self.num_envs, self.get_num_actors_per_env(), -1)
        self._stage_states = roots[:, self._stage_handles[0]]
        start = self._room_handles[0][0]
        self._room_states = roots[:, start:start + 6]
        self._room_centers += self._scene_origins[:, 2, None]
        self._wall_starts += self._scene_origins[:, 2, None, :2]
        self._wall_ends += self._scene_origins[:, 2, None, :2]
        self._foot_vertices = load_foot_vertices(cfg['env']['asset'], self.device)
        self._foot_body_ids = [self.gym.find_actor_rigid_body_handle(self.envs[0],
            self.humanoid_handles_all[0][0], name) for name in ('right_foot', 'left_foot')]
        self.physics_metadata = dict(version=1, terrain='separate_fixed_scenes',
            scene_spacing=cfg['env']['sceneSpacing'], rsi_foot_clearance=.005)
        self._scenario_metrics = torch.zeros(3, 5, device=self.device)
        self._scenario_tasks = torch.zeros(3, 4, 3, device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        distance = self.cfg['env']['sceneSpacing']
        if spacing < distance + self._room['size_range'][1] / 2 + self._room['wall_thickness']:
            raise ValueError('envSpacing must contain all three fixed scenes')
        self._scene_offsets = torch.tensor([[0., -distance, 0.], [0., 0., 0.], [0., distance, 0.]], device=self.device)
        self._bridge_stage_handles = []
        low, high = self._room['size_range']
        choices = np.arange(low, high + self._cliff['size_step'] / 2, self._cliff['size_step'])
        dimensions = choices[np.random.randint(len(choices), size=(num_envs, 2))]
        self._room_half_size = torch.tensor(dimensions / 2, device=self.device, dtype=torch.float)
        centers, half, self._wall_starts, self._wall_ends = room_walls(
            self._room_half_size, self._room['wall_thickness'],
            self._room['wall_height'], self._room['partition']['passage_width'])
        centers[..., 2] += self._room['floor_height']
        floor = torch.zeros(num_envs, 1, 3, device=self.device)
        floor[..., 2] = self._room['floor_height'] / 2
        floor_half = torch.cat((self._room_half_size, floor[:, 0, 2:]), -1)
        self._room_centers = torch.cat((floor, centers), 1)
        self._room_half = torch.cat((floor_half[:, None], half), 1)
        self._room_assets, self._room_handles = {}, []
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        if env_id == 0:
            need = self._bridge_half_size[:, 1] + self._bridge['landing_margin']
            self._fixed_far_offset = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * (
                self._stage_half_size[:, 1] + self._far_stage_half_size[:, 1] - 2 * need)
        super()._build_env(env_id, env_ptr, humanoid_asset)
        pose = gymapi.Transform()
        origin = self._scene_origin(env_ptr, 1)
        pose.p = gymapi.Vec3(origin.x, origin.y, origin.z + self._cliff['height'] / 2)
        handle = self.gym.create_actor(env_ptr, self._stage_assets[tuple(self._stage_dimensions[env_id])],
                                      pose, 'bridge_stage_a', env_id, 0)
        self._bridge_stage_handles.append(handle)
        self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.32, .36, .42))
        origin = self._scene_origin(env_ptr, 2)
        handles = []
        for index in range(6):
            size = tuple((2 * self._room_half[env_id, index]).cpu().tolist())
            if size not in self._room_assets:
                self._room_assets[size] = self.gym.create_box(self.sim, *size, self._stage_asset_options())
            pose = gymapi.Transform()
            center = self._room_centers[env_id, index].cpu().tolist()
            pose.p = gymapi.Vec3(origin.x + center[0], origin.y + center[1], origin.z + center[2])
            handle = self.gym.create_actor(env_ptr, self._room_assets[size], pose, 'mixed_room_' + str(index), env_id, 0)
            color = (0.32, 0.36, 0.42) if index == 0 else (0.52, 0.57, 0.64) if index == 5 else (0.74, 0.70, 0.62)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            handles.append(handle)
        self._room_handles.append(handles)

    def _scene_origin(self, env_ptr, scene):
        origin = self.gym.get_env_origin(env_ptr)
        return gymapi.Vec3(origin.x, origin.y + float(self._scene_offsets[scene, 1]), origin.z)

    def _stage_origin(self, env_id, env_ptr):
        return self._scene_origin(env_ptr, 0)

    def _far_stage_position(self, env_id, env_ptr):
        origin = self._scene_origin(env_ptr, 1)
        x = self._stage_half_size[env_id, 0] + self._bridge_gap[env_id] + self._far_stage_half_size[env_id, 0]
        return gymapi.Vec3(origin.x + float(x), origin.y + float(self._fixed_far_offset[env_id]),
                          origin.z + LIFT + self._cliff['height'] / 2)

    def _far_stage_offset(self, env_ids):
        return self._fixed_far_offset[env_ids]

    def _reset_envs(self, env_ids):
        if not len(env_ids):
            return
        self._scenario_ids[env_ids] = torch.multinomial(self._scenario_probabilities, len(env_ids), replacement=True)
        self._env_origins[env_ids] = self._scene_origins[env_ids, self._scenario_ids[env_ids]]
        self._floor_height[env_ids] = torch.where(self._scenario_ids[env_ids] == 2,
            self._room['floor_height'], self._cliff['height'])
        super()._reset_envs(env_ids)

    def _sample_bridge_active(self, env_ids):
        return self._scenario_ids[env_ids] == 1

    def _sample_scene(self, env_ids, logical_sizes):
        human, objects, bases = super()._sample_scene(env_ids, logical_sizes)
        env, agent = self._expand_slots(env_ids)
        assigned = agent_object_indices(self.relation_runtime.graph, env, agent).view(-1, 2)
        own_sizes = logical_sizes.gather(1, assigned[..., None].expand(-1, -1, 3))
        self._goal_xy[env_ids] = sample_stage_goals(self._stage_half_size[env_ids], own_sizes, self._cliff['spawn_margin'])
        room_rows = (self._scenario_ids[env_ids] == 2).nonzero().flatten()
        if len(room_rows):
            ids = env_ids[room_rows]
            kinds = classify_templates(self.relation_runtime.graph, env, agent, True).view(-1, 2)[room_rows]
            pickup_cross = ((kinds == 0) | (kinds == 3)) & (torch.rand(kinds.shape, device=self.device)
                < self._room['partition'].get('pickup_cross_probability', 0.))
            human[room_rows], objects[room_rows], self._goal_xy[ids] = sample_partition_layout(
                self._room_half_size[ids], logical_sizes[room_rows], assigned[room_rows],
                self._cliff['spawn_margin'], self._room['object_margin'],
                self._room['wall_thickness'], self._room['partition']['passage_width'], pickup_cross)
            bases[room_rows] = self._room['floor_height']
        kinds = classify_templates(self.relation_runtime.graph, env, agent, True)
        counts = torch.bincount(self._scenario_ids[env] * 4 + kinds, minlength=12).view(3, 4)
        self._scenario_tasks[..., 0] += counts
        return human, objects, bases

    def _place_humans(self, ids, xy, yaw):
        super()._place_humans(ids, xy, yaw)
        dz = self._floor_height[ids] - self._cliff['height']
        self._humanoid_root_states[ids, :, 2] += dz[:, None]
        self._kinematic_humanoid_rigid_body_states[ids, :, :, 2] += dz[:, None, None]
        lowest = lowest_foot_height(self._kinematic_humanoid_rigid_body_states[ids],
                                   self._foot_body_ids, self._foot_vertices)
        support = self._env_origins[ids, 2] + self._floor_height[ids]
        lift = (support[:, None] + self.physics_metadata['rsi_foot_clearance'] - lowest).clamp_min(0.)
        self._humanoid_root_states[ids, :, 2] += lift
        self._kinematic_humanoid_rigid_body_states[ids, :, :, 2] += lift[:, :, None]

    def _compute_clean_humanoid_nodes(self, body_pos, body_rot, body_vel, body_ang_vel, origins, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        local = body_pos.clone()
        local[..., 2] -= (self._floor_height[ids] + origins[:, 2])[:, None, None]
        return HumanoidMA._compute_clean_humanoid_nodes(self, local, body_rot, body_vel, body_ang_vel, origins)

    def _reset_task(self, env_ids):
        self._tar_pos[env_ids] = self._env_origins[env_ids, None]
        self._tar_pos[env_ids, :, 2] += self._floor_height[env_ids, None]
        room = env_ids[self._scenario_ids[env_ids] == 2]
        self._tar_pos[room, :, 1] += (self._room_half_size[room, 1] - self._room['partition']['passage_width'] / 2)[:, None]
        env, agent = self._expand_slots(env_ids)
        active = classify_templates(self.relation_runtime.graph, env, agent, True) == 3
        env, agent = env[active], agent[active]
        slots = agent_goal_indices(self.relation_runtime.graph, env, agent)
        boxes = self._scenario_physical_object(env, agent)
        self._tar_pos[env, slots, :2] = self._goal_xy[env, agent] + self._env_origins[env, :2]
        self._tar_pos[env, slots, 2] = self._env_origins[env, 2] + self._floor_height[env] + self._box_size[env, boxes, 2] / 2
        if self._enable_markers:
            for e, a, g in zip(env.tolist(), agent.tolist(), slots.tolist()):
                self.gym.set_rigid_body_color(self.envs[e], self._marker_handles[e * 2 + g], 0,
                                             gymapi.MESH_VISUAL, self._agent_color(a))

    def _sensor_boxes(self, ids):
        centers, quats, half = super()._sensor_boxes(ids)
        centers[:, 0, 2] = torch.where(self._scenario_ids[ids] == 2, HIDDEN, centers[:, 0, 2])
        room = self._room_states[ids, :, :3].clone()
        room[..., 2] = torch.where(self._scenario_ids[ids, None] == 2, room[..., 2], HIDDEN)
        return (torch.cat((centers, room), 1), torch.cat((quats, self._room_states[ids, :, 3:7]), 1),
                torch.cat((half, self._room_half[ids]), 1))

    def _terrain_height(self, xy):
        ground = super()._terrain_height(xy)
        shape = (-1,) + (1,) * (xy.ndim - 2)
        return torch.where((self._scenario_ids == 2).view(shape),
                           (self._env_origins[:, 2] + self._room['floor_height']).view(shape), ground)

    def _cpa_collision_risk(self, roots, min_dist):
        agents = super()._cpa_collision_risk(roots, min_dist)
        walls = compute_wall_cpa_collision_penalty(roots, self._humanoid_root_states[..., 7:10],
            self._wall_starts, self._wall_ends, min_dist, self.dt, self._wall_collision_ttc_discount)
        return torch.maximum(agents, walls * (self._scenario_ids == 2)[:, None])

    def _record_cliff_metrics(self, fallen):
        active = getattr(self, '_cliff_diagnostic_active', torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._cliff_diagnostic_active = active & (self._scenario_ids != 2)
        super()._record_cliff_metrics(fallen)
        self._cliff_diagnostic_active = active
        env, agent = self._expand_slots(torch.arange(self.num_envs, device=self.device))
        kinds = classify_templates(self.relation_runtime.graph, env, agent, True).view(-1, 2)
        for scene in range(3):
            selected = (self._scenario_ids == scene) & active
            ending = selected & self.reset_buf.bool()
            self._scenario_metrics[scene] += torch.stack((selected.sum(), ending.sum(),
                (fallen & selected[:, None]).sum(), (self.progress_buf * ending).sum(),
                (ending & ~fallen.any(-1)).sum()))
            for kind in range(4):
                mask = selected[:, None] & (kinds == kind)
                self._scenario_tasks[scene, kind, 1] += mask.sum()
                self._scenario_tasks[scene, kind, 2] += (self.relation_runtime.done & mask).sum()

    def cliff_diagnostics(self, reset=False):
        result = super().cliff_diagnostics(reset)
        for name, row, tasks in zip(SCENARIOS, self._scenario_metrics, self._scenario_tasks):
            for key, value in zip(('env_steps', 'episodes', 'agent_falls', 'episode_steps', 'timeouts'), row):
                result['scenario/' + name + '/' + key] = value.item()
            for task, counts in zip(TASKS, tasks):
                for key, value in zip(('sampled_agents', 'agent_steps', 'goal_success_steps'), counts):
                    result['scenario/' + name + '/' + task + '/' + key] = value.item()
        if reset:
            self._scenario_metrics.zero_()
            self._scenario_tasks.zero_()
        return result

    def _draw_ontop_context(self):
        super()._draw_ontop_context()
        env = torch.zeros(2, dtype=torch.long, device=self.device)
        agents = torch.arange(2, device=self.device)
        active = classify_templates(self.relation_runtime.graph, env, agents, True) == 3
        slots = agent_goal_indices(self.relation_runtime.graph, env, agents)
        angle = np.linspace(0, 2 * np.pi, 33)
        origin = self.gym.get_env_origin(self.envs[0])
        origin = self._env_origins.new_tensor([origin.x, origin.y, origin.z])
        for agent in agents[active].tolist():
            goal = (self._tar_pos[0, slots[agent]] - origin).cpu().numpy()
            points = np.stack((goal[0] + .3 * np.cos(angle), goal[1] + .3 * np.sin(angle),
                               np.full_like(angle, self._floor_height[0].item() + .025)), -1)
            lines = np.stack((points[:-1], points[1:]), 1).astype(np.float32)
            color = self._agent_color(agent)
            self.gym.add_lines(self.viewer, self.envs[0], len(lines), lines,
                               np.tile([color.x, color.y, color.z], (32, 1)).astype(np.float32))
