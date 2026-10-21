import copy
import os
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.bridge_geometry import (
    bridge_control, bridge_schedule, bridge_trajectory, compute_agent_cpa_collision_penalty,
    sample_bridge_layout)
from env.tasks.multi_agent.vision_sensor import vertical_height

LIFT = 0.001
HIDDEN = -1000.
METRICS = ('agent_steps', 'before_arrival_steps', 'waiting_near_gap_steps', 'falls_before_arrival',
           'falls_after_arrival', 'far_stage_steps', 'episodes', 'arrived_episodes', 'crossed_agents',
           'approach_episodes', 'approach_started', 'approach_arrived')


class HumanoidMABridge(HumanoidMACliff):
    _fixed_far_stage = False

    def __init__(self, cfg, *args, **kwargs):
        self._bridge = cfg['env']['bridge']
        self._bridge_friction = float(os.environ.get('BRIDGE_FRICTION', '1.0'))
        if not np.isfinite(self._bridge_friction) or self._bridge_friction < 0:
            raise ValueError('BRIDGE_FRICTION must be finite and nonnegative')
        self._agent_collision_ttc_discount = cfg['env']['agentCollisionTTCDiscount']
        if not 0 < self._agent_collision_ttc_discount <= 1:
            raise ValueError('agentCollisionTTCDiscount must be in (0, 1]')
        if cfg['env']['cliff'].get('sensor_occlusion') != 'none':
            raise ValueError('Bridge scenes use the vertical height sensor without occlusion')
        metadata = copy.deepcopy(self._bridge)
        self._bridge_probability = float(os.environ.get('BRIDGE_PROBABILITY', self._bridge.get('probability', .5)))
        super().__init__(cfg, *args, **kwargs)
        self.sensor_metadata.update(version=4, bridge=metadata)
        n, actors = self.num_envs, self.get_num_actors_per_env()
        roots = self._root_states.view(n, actors, -1)
        base = actors * torch.arange(n, device=self.device, dtype=torch.int32)
        self._far_stage_states = roots[:, self._far_stage_handle]
        self._bridge_states = roots[:, self._bridge_handle]
        self._bridge_actor_ids = torch.cat((base + self._far_stage_handle, base + self._bridge_handle))
        self._bridge_body_ids = torch.tensor([self.gym.get_actor_rigid_body_index(
            env, self._bridge_handle, 0, gymapi.DOMAIN_SIM) for env in self.envs], device=self.device)
        self._bridge_forces = torch.zeros(self._rigid_body_state.shape[0], 3, device=self.device)
        self._bridge_torques = torch.zeros_like(self._bridge_forces)
        self._bridge_mass = self._bridge['density'] * (2 * self._bridge_half_size).prod(-1)
        self._bridge_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._bridge_present = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._bridge_target = torch.zeros(n, 3, device=self.device)
        self._bridge_start = torch.zeros(n, 3, device=self.device)
        self._bridge_goal = torch.zeros(n, 3, device=self.device)
        self._bridge_wait = torch.zeros(n, device=self.device)
        self._bridge_duration = torch.ones(n, device=self.device)
        self._bridge_crossed = torch.zeros(n, 2, dtype=torch.bool, device=self.device)
        self._bridge_metrics = torch.zeros(len(METRICS), device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        b, step, height = self._bridge, self._bridge['size_step'], self._cliff['height']
        high_a = self._cliff['size_range'][1]
        low, high = b['far_size_range']
        sizes = np.arange(low, high + self._cliff['size_step'] / 2, self._cliff['size_step'])
        if spacing < high_a / 2 + high + max(b['gap_range'][1], b['width_range'][1] / 2 + b['approach_distance'][1]):
            raise ValueError('envSpacing is too small for two stages and the approaching bridge')
        far = sizes[np.random.randint(len(sizes), size=(num_envs, 2))]
        gap = np.round(np.random.uniform(*b['gap_range'], num_envs) / step) * step
        width = np.round(np.random.uniform(*b['width_range'], num_envs) / step) * step
        bridge = np.round(np.stack((gap - 2 * b['clearance'], width,
                                    np.full(num_envs, height - b['top_drop'])), -1), 4)
        self._far_stage_dimensions, self._bridge_dimensions = far, bridge
        self._far_stage_half_size = torch.tensor(far / 2, device=self.device, dtype=torch.float)
        self._bridge_gap = torch.tensor(gap, device=self.device, dtype=torch.float)
        self._bridge_half_size = torch.tensor(bridge / 2, device=self.device, dtype=torch.float)
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.max_angular_velocity = 0.
        options.density = b['stage_density']
        options.fix_base_link = self._fixed_far_stage
        self._far_stage_assets = {key: self.gym.create_box(self.sim, float(key[0]), float(key[1]), height, options)
                                  for key in set(map(tuple, far))}
        options.density = b['density']
        options.fix_base_link = False
        self._bridge_assets = {key: self.gym.create_box(self.sim, *map(float, key), options)
                               for key in set(map(tuple, bridge))}
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        origin = self.gym.get_env_origin(env_ptr)
        pose = gymapi.Transform()
        pose.p = self._far_stage_position(env_id, env_ptr)
        asset = self._far_stage_assets[tuple(self._far_stage_dimensions[env_id])]
        self._far_stage_handle = self.gym.create_actor(env_ptr, asset, pose, 'far_stage', env_id, 0)
        self.gym.set_rigid_body_color(env_ptr, self._far_stage_handle, 0, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(0.32, 0.36, 0.42))
        pose.p = gymapi.Vec3(origin.x, origin.y, origin.z + self._bridge['parked_height'] + 5)
        asset = self._bridge_assets[tuple(self._bridge_dimensions[env_id])]
        self._bridge_handle = self.gym.create_actor(env_ptr, asset, pose, 'bridge', env_id, 0)
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self._bridge_handle)
        for prop in shape_props:
            prop.friction = self._bridge_friction
        self.gym.set_actor_rigid_shape_properties(env_ptr, self._bridge_handle, shape_props)
        self.gym.set_rigid_body_color(env_ptr, self._bridge_handle, 0, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(0.62, 0.45, 0.25))

    def _far_stage_position(self, env_id, env_ptr):
        origin = self.gym.get_env_origin(env_ptr)
        return gymapi.Vec3(origin.x, origin.y, origin.z + self._bridge['parked_height'])

    def _far_stage_offset(self, env_ids):
        return None

    def _near_start_probability(self):
        spec = self._bridge['near_start']
        fraction = 1. if self._is_eval else min(
            getattr(self, '_hard_skill_training_step', 0) / spec['anneal_steps'], 1.)
        return spec['probability_start'] + fraction * (spec['probability_end'] - spec['probability_start'])

    def _sample_bridge_active(self, env_ids):
        return torch.rand(len(env_ids), device=self.device) < self._bridge_probability

    def _sample_scene(self, env_ids, logical_sizes):
        human_xy, object_xy, object_base = super()._sample_scene(env_ids, logical_sizes)
        n, b, height = len(env_ids), self._bridge, self._cliff['height']
        active = self._sample_bridge_active(env_ids)
        self._bridge_active[env_ids] = active
        self._bridge_present[env_ids] = False
        self._bridge_crossed[env_ids] = False
        origins = self._env_origins[env_ids]
        far = torch.zeros(n, 13, device=self.device)
        far[:, :3] = origins
        far[:, 2] += b['parked_height']
        far[:, 6] = 1
        bridge = far.clone()
        bridge[:, 2] += 5
        rows = active.nonzero().flatten()
        if len(rows):
            ids, origin = env_ids[rows], origins[rows]
            layout = sample_bridge_layout(self._stage_half_size[ids], self._far_stage_half_size[ids],
                self._bridge_gap[ids], self._bridge_half_size[ids, :2], logical_sizes[rows], b,
                self._cliff['spawn_margin'], self._near_start_probability(),
                far_offset=self._far_stage_offset(ids))
            human_xy[rows], object_xy[rows] = layout['humans'], layout['objects']
            object_base[rows] = height
            far[rows, :2] = layout['far_stage'] + origin[:, :2]
            far[rows, 2] = origin[:, 2] + LIFT + height / 2
            goal = torch.cat((layout['landing'] + origin[:, :2],
                              (origin[:, 2] + LIFT + self._bridge_half_size[ids, 2])[:, None]), -1)
            start = goal.clone()
            present = torch.rand(len(rows), device=self.device) < b['present_probability']
            start[~present, :2] = layout['start'][~present] + origin[~present, :2]
            low, high = b['arrival_time_range']
            arrival = low * (high / low) ** torch.rand(len(rows), device=self.device)
            duration, wait = bridge_schedule((goal - start).norm(dim=-1), arrival, b['speed_range'], self.dt)
            wait = torch.where(present, 0., wait)
            self._bridge_start[ids], self._bridge_goal[ids] = start, goal
            self._bridge_present[ids] = present
            self._bridge_target[ids] = torch.cat((layout['goal'] + origin[:, :2],
                                                  (origin[:, 2] + LIFT + height)[:, None]), -1)
            self._bridge_wait[ids], self._bridge_duration[ids] = wait, duration
            bridge[rows, :3], bridge[rows, 7:10] = bridge_trajectory(start, goal, -wait, duration)
        if not self._fixed_far_stage:
            self._far_stage_states[env_ids] = far
        self._bridge_states[env_ids] = bridge
        return human_xy, object_xy, object_base

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        # Goals are inactive (no AT edges) and default to the env origin inside stage A;
        # bridge episodes keep them on the far stage top near its gap edge instead.
        ids = env_ids[self._bridge_active[env_ids]]
        self._tar_pos[ids] = self._bridge_target[ids, None].expand(-1, self.num_agents, -1)

    def _extra_reset_actor_ids(self, env_ids):
        if self._fixed_far_stage:
            return [self._bridge_actor_ids.view(2, -1)[1, env_ids]]
        return [self._bridge_actor_ids.view(2, -1)[:, env_ids].flatten()]

    def _compute_reward(self, actions):
        return self._compute_relation_reward(self._cpa_collision_risk)

    def _cpa_collision_risk(self, roots, min_dist):
        return compute_agent_cpa_collision_penalty(
            roots, self._humanoid_root_states[..., 7:10], min_dist, self.dt,
            self._agent_collision_ttc_discount)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        b = self._bridge
        elapsed = (self.progress_buf + 1) * self.dt - self._bridge_wait
        target, velocity = bridge_trajectory(self._bridge_start, self._bridge_goal, elapsed, self._bridge_duration)
        force = bridge_control(target, velocity, self._bridge_states, self._bridge_mass,
                               b['tracking_frequency'], b['damping_ratio'], b['max_acceleration'])
        self._bridge_forces[self._bridge_body_ids] = force * self._bridge_active[:, None]

    def _physics_step(self):
        forces = gymtorch.unwrap_tensor(self._bridge_forces)
        torques = gymtorch.unwrap_tensor(self._bridge_torques)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.apply_rigid_body_force_tensors(self.sim, forces, torques, gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)

    def _bridge_terrain(self, ids):
        states = torch.stack((self._far_stage_states[ids], self._bridge_states[ids]), 1)
        centers = states[..., :3].clone()
        centers[..., 2] = torch.where(self._bridge_active[ids, None], centers[..., 2], HIDDEN)
        far_half = torch.cat((self._far_stage_half_size[ids],
                              torch.full_like(self._far_stage_half_size[ids][:, :1], self._cliff['height'] / 2)), -1)
        return centers, states[..., 3:7], torch.stack((far_half, self._bridge_half_size[ids]), 1)

    def _sensor_boxes(self, ids):
        boxes = super()._sensor_boxes(ids)
        return tuple(torch.cat(pair, 1) for pair in zip(boxes, self._bridge_terrain(ids)))

    def _terrain_height(self, xy):
        centers, quaternions, half_sizes = self._bridge_terrain(slice(None))
        return torch.maximum(super()._terrain_height(xy), vertical_height(
            xy, centers, quaternions, half_sizes, self._env_origins[:, 2]))

    def _bridge_arrived(self):
        return self.progress_buf * self.dt >= self._bridge_wait + self._bridge_duration

    def _record_cliff_metrics(self, fallen):
        super()._record_cliff_metrics(fallen)
        root = self._rigid_body_pos[..., 0, :] - self._env_origins[:, None]
        edge = self._stage_half_size[:, None, 0]
        on_near_stage = (root[..., :2].abs() <= self._stage_half_size[:, None]).all(-1)
        near_gap = on_near_stage & (edge - root[..., 0] < 1.)
        far = (root[..., 0] > edge + self._bridge_gap[:, None]) & (root[..., 2] > self._cliff['height'] + .6)
        stopped = self._rigid_body_vel[..., 0, :2].norm(dim=-1) < .15
        arrived = self._bridge_arrived()
        active = getattr(self, '_cliff_diagnostic_active',
                         torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        bridge = (self._bridge_active & active)[:, None].expand(-1, 2)
        waiting = bridge & ~arrived[:, None]
        self._bridge_crossed |= far & bridge
        ending = self.reset_buf.bool() & self._bridge_active & active
        approach = ending & ~self._bridge_present
        started = self.progress_buf * self.dt > self._bridge_wait
        self._bridge_metrics += torch.stack((
            bridge.sum(), waiting.sum(), (waiting & near_gap & stopped).sum(), (fallen & waiting).sum(),
            (fallen & bridge & arrived[:, None]).sum(), (far & bridge).sum(), ending.sum(),
            (ending & arrived).sum(), (self._bridge_crossed & ending[:, None]).sum(),
            approach.sum(), (approach & started).sum(), (approach & arrived).sum())).float()

    def cliff_diagnostics(self, reset=False):
        result = super().cliff_diagnostics(reset)
        result.update({'bridge/' + key: value.item() for key, value in zip(METRICS, self._bridge_metrics)})
        if reset:
            self._bridge_metrics.zero_()
        return result
