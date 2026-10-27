import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent.humanoid_ma_before import HumanoidMABefore
from env.tasks.multi_agent.bridge_geometry import (
    bridge_control, bridge_schedule, bridge_trajectory, sample_bridge_layout, stage_height,
    terrain_fall, vertical_height)
from utils import torch_utils
from utils.bridge_spec import sample_templates, token_mapping
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import copy_graph_rows, permute_graph

LIFT = 0.001
METRICS = ('agent_steps', 'before_arrival_steps', 'waiting_near_gap_steps', 'falls_before_arrival',
           'falls_after_arrival', 'far_stage_steps', 'episodes', 'arrived_episodes', 'crossed_agents',
           'approach_episodes', 'approach_started', 'approach_arrived')


class HumanoidMABridge(HumanoidMABefore):
    """Two stages across a gap; a force-driven bridge box (an edge-less object token) closes it."""
    _extra_slots = 1

    def __init__(self, cfg, *args, **kwargs):
        self._bridge_cfg = cfg['env']['bridge']
        self._amp_num_families = 4
        self._bridge_training_step = 0
        super().__init__(cfg, *args, **kwargs)

    def _sample_bridge_geometry(self, n):
        # Returns gap width, bridge box size and stage height per environment.
        b, step = self._bridge_cfg, self._bridge_cfg['size_step']

        def draw(bounds):
            return torch.round((bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, device=self.device)) / step) * step

        gap = draw(b['gap_range'])
        size = torch.stack((gap - 2 * b['clearance'], draw(b['width_range']),
                            torch.full((n,), b['height'] - b['top_drop'], device=self.device)), -1)
        return gap, torch.round(size * 1e4) / 1e4, torch.full((n,), float(b['height']), device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        b = self._bridge_cfg
        self._bridge_slot = self.num_physical_objects - self._extra_slots
        self._bridge_gap, self._bridge_size, self._stage_height = self._sample_bridge_geometry(num_envs)
        step = b['stage_size_step']
        near = np.arange(b['stage_size_range'][0], b['stage_size_range'][1] + step / 2, step)
        far = np.arange(b['far_size_range'][0], b['far_size_range'][1] + step / 2, step)
        height = self._stage_height.cpu().numpy()[:, None]
        self._stage_dimensions = np.concatenate((near[np.random.randint(len(near), size=(num_envs, 2))], height), -1)
        self._far_stage_dimensions = np.concatenate((far[np.random.randint(len(far), size=(num_envs, 2))], height), -1)
        self._stage_half_size = torch.tensor(self._stage_dimensions[:, :2] / 2, device=self.device, dtype=torch.float)
        self._far_stage_half_size = torch.tensor(self._far_stage_dimensions[:, :2] / 2, device=self.device, dtype=torch.float)
        options = gymapi.AssetOptions()
        options.fix_base_link = b['mode'] != 'push4'
        options.disable_gravity = True
        if b['mode'] == 'push4':
            options.max_angular_velocity = 0.
            options.density = b['stage_density']
        self._stage_assets = {key: self.gym.create_box(self.sim, *map(float, key), options)
                              for key in set(map(tuple, self._stage_dimensions))}
        # The far stage is a gravity-free heavy body so that resets can move it.
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.max_angular_velocity = 0.
        options.density = b['stage_density']
        self._far_stage_assets = {key: self.gym.create_box(self.sim, *map(float, key), options)
                                  for key in set(map(tuple, self._far_stage_dimensions))}
        super()._create_envs(num_envs, spacing, num_per_row)

    def _load_box_asset(self):
        super()._load_box_asset()
        self._box_size[:, self._bridge_slot] = self._bridge_size
        self._create_bridge_assets()
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)

    def _create_bridge_assets(self):
        b, n, p, slot = self._bridge_cfg, self.num_envs, self.num_physical_objects, self._bridge_slot
        self._box_density.view(n, p)[:, slot] = b['density']
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.max_angular_velocity = 0.
        options.density = b['density']
        sizes = [tuple(row) for row in self._bridge_size.cpu().tolist()]
        assets = {key: self.gym.create_box(self.sim, *key, options) for key in set(sizes)}
        for env, key in enumerate(sizes):
            self._box_assets[env * p + slot] = assets[key]

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        if object_id != self._bridge_slot:
            return
        handle = self._box_handles[-1]
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        for prop in props:
            prop.friction = self._bridge_cfg['friction']
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        origin = self.gym.get_env_origin(env_ptr)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(origin.x, origin.y, origin.z + float(self._stage_height[env_id]) / 2)
        asset = self._stage_assets[tuple(self._stage_dimensions[env_id])]
        self._stage_handle = self.gym.create_actor(env_ptr, asset, pose, 'near_stage', env_id, 0)
        self._near_stage_handle = self._stage_handle
        pose.p = gymapi.Vec3(origin.x, origin.y, origin.z + self._bridge_cfg['parked_height'])
        asset = self._far_stage_assets[tuple(self._far_stage_dimensions[env_id])]
        self._far_stage_handle = self.gym.create_actor(env_ptr, asset, pose, 'far_stage', env_id, 0)
        for handle in (self._stage_handle, self._far_stage_handle):
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.32, 0.36, 0.42))

    def _build_platform_state_tensors(self):
        super()._build_platform_state_tensors()
        n, m, p = self.num_envs, self.num_agents, self.num_physical_objects
        actors = self.get_num_actors_per_env()
        roots = self._root_states.view(n, actors, -1)
        self._far_stage_states = roots[:, self._far_stage_handle]
        self._far_stage_actor_ids = actors * torch.arange(n, device=self.device, dtype=torch.int32) + self._far_stage_handle
        self._bridge_states = self._box_states[:, self._bridge_slot]
        self._bridge_body_ids = torch.tensor([self.gym.get_actor_rigid_body_index(
            env, self._box_handles[i * p + self._bridge_slot], 0, gymapi.DOMAIN_SIM) for i, env in enumerate(self.envs)],
            device=self.device, dtype=torch.long)
        self._bridge_forces = torch.zeros(self._rigid_body_state.shape[0], 3, device=self.device)
        self._bridge_torques = torch.zeros_like(self._bridge_forces)
        self._bridge_mass = self._box_density.view(n, p)[:, self._bridge_slot] * self._bridge_size.prod(-1)
        self._bridge_present = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._bridge_start = torch.zeros(n, 3, device=self.device)
        self._bridge_goal = torch.zeros(n, 3, device=self.device)
        self._bridge_target = torch.zeros(n, 3, device=self.device)
        self._bridge_wait = torch.zeros(n, device=self.device)
        self._bridge_duration = torch.ones(n, device=self.device)
        self._bridge_crossed = torch.zeros(n, m, dtype=torch.bool, device=self.device)
        self._floor_start = torch.zeros(n, m, dtype=torch.bool, device=self.device)
        self._crosser = torch.ones(n, m, dtype=torch.bool, device=self.device)
        self._bridge_metrics = torch.zeros(len(METRICS), device=self.device)

    def get_relation_suffix_size(self):
        return 6 * self.num_agents

    def _compute_clean_scene_obs(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        return torch.cat((super()._compute_clean_scene_obs(env_ids), self._before_predecessor[ids].float()), -1)

    def _near_start_probability(self):
        spec = self._bridge_cfg['near_start']
        fraction = 1. if self._is_eval else min(self._bridge_training_step / spec['anneal_steps'], 1.)
        return spec['probability_start'] + fraction * (spec['probability_end'] - spec['probability_start'])

    def _sample_episode_graph(self, ids):
        k, m = len(ids), self.num_agents
        templates = sample_templates(k, self._task_graph_preset, self._bridge_cfg['task_probabilities'], self.device)
        primary = torch.arange(m, device=self.device)[None] + templates * m
        self._templates[ids], self._before_predecessor[ids] = templates, -1
        self._before_distance_modes[ids], self._before_near_pairs[ids] = 0, False
        self._agent_box_assignment[ids] = primary
        self._logical_box_order[ids] = token_mapping(primary, self._bridge_slot, 4 * m)
        self._task_token_count[ids] = m
        graph = compose_canonical_graph(templates, num_objects=self.num_objects)
        self._finish_episode_graph(ids, graph, templates)

    def _finish_episode_graph(self, ids, graph, templates):
        if self._relation_graph_spec['shuffle_edge_order']:
            graph = permute_graph(graph, torch.rand(len(ids), 2 * self.num_agents, device=self.device).argsort(-1))
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._update_box_assignment_colors(ids)
        self._color_bridges(ids)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:6] += torch.bincount(templates.flatten(), minlength=5)

    def _color_bridges(self, ids):
        rendered = ids.tolist() if self.viewer is not None else [0] if self._video_enabled and (ids == 0).any() else []
        p = self.num_physical_objects
        for env in rendered:
            self.gym.set_rigid_body_color(self.envs[env], self._box_handles[env * p + self._bridge_slot], 0,
                                         gymapi.MESH_VISUAL, gymapi.Vec3(0.62, 0.45, 0.25))

    def _far_stage_layout(self, ids, boxes, bridge_half):
        # Stage B, the landing site and the task boxes on stage B; humans as drawn for stage A.
        b, k = self._bridge_cfg, len(ids)
        height, origin = self._stage_height[ids], self._env_origins[ids]
        sizes = self._box_size[ids[:, None], boxes]
        layout = sample_bridge_layout(self._stage_half_size[ids], self._far_stage_half_size[ids],
                                      self._bridge_gap[ids], bridge_half[:, :2], sizes, b,
                                      b['spawn_margin'], self._near_start_probability())
        state = torch.zeros(k, boxes.shape[1], 13, device=self.device)
        state[..., :2] = layout['objects'] + origin[:, None, :2]
        state[..., 2] = origin[:, None, 2] + height[:, None] + sizes[..., 2] / 2 + 0.01
        axis = torch.tensor([0., 0., 1.], device=self.device).expand(state.shape[0] * state.shape[1], -1)
        yaw = torch.rand(state.shape[0] * state.shape[1], device=self.device) * 2 * torch.pi
        state[..., 3:7] = quat_from_angle_axis(yaw, axis).view(k, -1, 4)
        self._box_states[ids[:, None], boxes] = state
        far = torch.zeros(k, 13, device=self.device)
        far[:, :2] = layout['far_stage'] + origin[:, :2]
        far[:, 2] = origin[:, 2] + LIFT + height / 2
        far[:, 6] = 1
        self._far_stage_states[ids] = far
        self._bridge_target[ids] = torch.cat((layout['goal'] + origin[:, :2], (origin[:, 2] + LIFT + height)[:, None]), -1)
        self._bridge_crossed[ids] = False
        return layout, state

    def _reset_boxes(self, ids):
        self._reset_all_boxes_random_arena(ids)
        self._platform_pos[ids] = self._platform_default_pos[ids]
        b, k, m = self._bridge_cfg, len(ids), self.num_agents
        origin = self._env_origins[ids]
        mapping = self._logical_box_order[ids]
        distractor = torch.where(mapping[:, 2] == self._bridge_slot, mapping[:, 3], mapping[:, 2])
        boxes = torch.cat((mapping[:, :m], distractor[:, None]), -1)
        bridge_half = self._bridge_size[ids] / 2
        layout, state = self._far_stage_layout(ids, boxes, bridge_half)

        goal = torch.cat((layout['landing'] + origin[:, :2], (origin[:, 2] + LIFT + bridge_half[:, 2])[:, None]), -1)
        start = goal.clone()
        present = torch.rand(k, device=self.device) < b['present_probability']
        start[~present, :2] = layout['start'][~present] + origin[~present, :2]
        low, high = b['arrival_time_range']
        arrival = low * (high / low) ** torch.rand(k, device=self.device)
        duration, wait = bridge_schedule((goal - start).norm(dim=-1), arrival, b['speed_range'], self.dt)
        wait = torch.where(present, 0., wait)
        self._bridge_start[ids], self._bridge_goal[ids], self._bridge_present[ids] = start, goal, present
        self._bridge_wait[ids], self._bridge_duration[ids] = wait, duration
        bridge = torch.zeros(k, 13, device=self.device)
        bridge[:, :3], bridge[:, 7:10] = bridge_trajectory(start, goal, -wait, duration)
        bridge[:, 6] = 1
        self._bridge_states[ids] = bridge

        target = state[torch.arange(k, device=self.device)[:, None], torch.arange(m, device=self.device)[None], :2]
        human = layout['humans'] + origin[:, None, :2]
        self._place_humans(ids, human, self._face(human, target), self._stage_height[ids, None].expand(-1, m))

    def _face(self, xy, target):
        direction = target - xy
        yaw = torch.atan2(direction[..., 1], direction[..., 0])
        return yaw + (torch.rand_like(yaw) * 2 - 1) * self._bridge_cfg['heading_jitter']

    def _place_humans(self, ids, xy, yaw, lift):
        k, m, nb = len(ids), self.num_agents, self.num_bodies
        root = self._humanoid_root_states[ids].clone()
        body = self._kinematic_humanoid_rigid_body_states[ids].clone()
        old_heading = torch_utils.calc_heading(root[..., 3:7].reshape(-1, 4)).view_as(yaw)
        axis = torch.tensor([0., 0., 1.], device=self.device).expand(yaw.numel(), -1)
        q = quat_from_angle_axis((yaw - old_heading).flatten(), axis).view(k, m, 4)
        qb = q[:, :, None].expand(-1, -1, nb, -1).reshape(-1, 4)
        offset = body[..., :3] - root[..., None, :3]
        root[..., :2] = xy
        root[..., 2] += lift
        root[..., 3:7] = quat_mul(q.reshape(-1, 4), root[..., 3:7].reshape(-1, 4)).view(k, m, 4)
        for start in (7, 10):
            root[..., start:start + 3] = quat_rotate(q.reshape(-1, 4), root[..., start:start + 3].reshape(-1, 3)).view(k, m, 3)
            body[..., start:start + 3] = quat_rotate(qb, body[..., start:start + 3].reshape(-1, 3)).view(k, m, nb, 3)
        body[..., :3] = quat_rotate(qb, offset.reshape(-1, 3)).view_as(offset) + root[..., None, :3]
        body[..., 3:7] = quat_mul(qb, body[..., 3:7].reshape(-1, 4)).view(k, m, nb, 4)
        self._humanoid_root_states[ids] = root
        self._kinematic_humanoid_rigid_body_states[ids] = body

    def _reset_task(self, ids):
        super()._reset_task(ids)
        # Goals without edges stay on the far stage near its gap edge.
        self._tar_pos[ids] = self._bridge_target[ids, None].expand(-1, self.num_agents, -1)

    def _configure_ontop_targets(self, ids):
        return

    def _ontop_scene_infeasible(self, ids):
        return torch.zeros(len(ids), dtype=torch.bool, device=self.device)

    def _extra_reset_actor_ids(self, env_ids):
        return [self._far_stage_actor_ids[env_ids]]

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        b = self._bridge_cfg
        elapsed = (self.progress_buf + 1) * self.dt - self._bridge_wait
        target, velocity = bridge_trajectory(self._bridge_start, self._bridge_goal, elapsed, self._bridge_duration)
        self._bridge_forces[self._bridge_body_ids] = bridge_control(
            target, velocity, self._bridge_states, self._bridge_mass,
            b['tracking_frequency'], b['damping_ratio'], b['max_acceleration'])

    def _physics_step(self):
        forces = gymtorch.unwrap_tensor(self._bridge_forces)
        torques = gymtorch.unwrap_tensor(self._bridge_torques)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.apply_rigid_body_force_tensors(self.sim, forces, torques, gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)

    def _terrain_height(self, xy, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        origins, height = self._env_origins[ids], self._stage_height[ids]
        states = torch.stack((self._far_stage_states[ids], self._bridge_states[ids]), 1)
        far_half = torch.cat((self._far_stage_half_size[ids], height[:, None] / 2), -1)
        half = torch.stack((far_half, self._bridge_size[ids] / 2), 1)
        near = stage_height(xy, origins, self._stage_half_size[ids], height)
        return torch.maximum(near, vertical_height(xy, states[..., :3], states[..., 3:7], half, origins[:, 2]))

    def _compute_clean_humanoid_nodes(self, body_pos, body_rot, body_vel, body_ang_vel, env_origins, env_ids=None):
        # Root height is measured from the surface under each human, as on flat Stage1 ground.
        support = self._terrain_height(body_pos[..., 0, :2], env_ids)
        body_pos = body_pos.clone()
        body_pos[..., 2] -= support[..., None]
        return super()._compute_clean_humanoid_nodes(body_pos, body_rot, body_vel, body_ang_vel, env_origins)

    def _compute_reset(self):
        ground = self._terrain_height(self._rigid_body_pos[..., :2])
        fallen = terrain_fall(self._contact_forces, self._rigid_body_pos, self._contact_body_ids, ground,
                              self._termination_heights, self.progress_buf, self._env_origins[:, 2],
                              self._floor_start)
        if not self._enable_early_termination:
            fallen.zero_()
        self._terminate_buf[:] = fallen.flatten().long()
        self.reset_buf[:] = (fallen.any(-1) | (self.progress_buf >= self.max_episode_length - 1)).long()
        self._record_bridge_metrics(fallen)

    def _bridge_arrived(self):
        return self.progress_buf * self.dt >= self._bridge_wait + self._bridge_duration

    def _record_bridge_metrics(self, fallen):
        root = self._rigid_body_pos[..., 0, :] - self._env_origins[:, None]
        edge = self._stage_half_size[:, None, 0]
        on_near_stage = (root[..., :2].abs() <= self._stage_half_size[:, None]).all(-1)
        near_gap = on_near_stage & (edge - root[..., 0] < 1.)
        far = (root[..., 0] > edge + self._bridge_gap[:, None]) & (root[..., 2] > self._stage_height[:, None] + .6)
        stopped = self._rigid_body_vel[..., 0, :2].norm(dim=-1) < .15
        arrived = self._bridge_arrived()
        crosser = self._crosser
        waiting = ~arrived[:, None] & crosser
        self._bridge_crossed |= far & crosser
        ending = self.reset_buf.bool()
        approach = ending & ~self._bridge_present
        started = self.progress_buf * self.dt > self._bridge_wait
        self._bridge_metrics += torch.stack((
            crosser.sum(), waiting.sum(), (waiting & near_gap & stopped).sum(), (fallen & waiting).sum(),
            (fallen & arrived[:, None] & crosser).sum(), (far & crosser).sum(), ending.sum(),
            (ending & arrived).sum(), (self._bridge_crossed & ending[:, None]).sum(), approach.sum(),
            (approach & started).sum(), (approach & arrived).sum())).float()

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        s = dict(zip(METRICS, self._bridge_metrics.clone()))
        agents = (s['episodes'] * self._crosser[0].sum()).clamp_min(1)
        result.update({
            'bridge/waiting_near_gap_fraction': s['waiting_near_gap_steps'] / s['before_arrival_steps'].clamp_min(1),
            'bridge/falls_before_arrival_per_agent': s['falls_before_arrival'] / agents,
            'bridge/falls_after_arrival_per_agent': s['falls_after_arrival'] / agents,
            'bridge/crossed_agent_fraction': s['crossed_agents'] / agents,
            'bridge/far_stage_step_fraction': s['far_stage_steps'] / s['agent_steps'].clamp_min(1),
            'bridge/arrived_episode_fraction': s['arrived_episodes'] / s['episodes'].clamp_min(1),
            'bridge/approach_arrived_fraction': s['approach_arrived'] / s['approach_episodes'].clamp_min(1),
            'bridge/episodes': s['episodes']})
        self._bridge_metrics.zero_()
        return result
