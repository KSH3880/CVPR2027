import copy
import os
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.cliff_geometry import (
    height_observation, sample_layout, sensor_grid, sensor_positions, stage_height, terrain_fall)
from env.tasks.multi_agent.vision_sensor import uniform_fan_grid, fan_observation, fan_height_observation
from utils import torch_utils
from utils.edge_scenario_spec import agent_object_indices, classify_templates


class HumanoidMACliff(HumanoidMACarry):
    _supports_holding_at = False

    def __init__(self, cfg, *args, **kwargs):
        self._cliff = cfg['env']['cliff']
        pattern = self._cliff.get('sensor_pattern', 'grid')
        self._square_sensor = pattern == 'square'
        # The square sensor reuses the fan's box height lookup, viewer cache and metrics.
        self._fan_sensor = pattern in ('fan', 'square')
        self._fan_view_data = None
        if self._square_sensor:
            if not (0 < self._cliff['sensor_range'] and self._cliff['sensor_resolution'] >= 2
                    and self._cliff['sensor_surfaces'] == 'stage_floor_boxes'
                    and self._cliff['sensor_occlusion'] == 'none'):
                raise ValueError('Unsupported square sensor configuration')
        elif self._fan_sensor:
            if not (0 < self._cliff['sensor_range']
                    and 0 < self._cliff['sensor_fov'] <= 180
                    and self._cliff['sensor_sampling'] == 'uniform_rings'
                    and self._cliff['sensor_resolution'] >= 2):
                raise ValueError('Invalid fan sensor sampling configuration')
            occlusion = self._cliff['sensor_occlusion']
            if (self._cliff['sensor_surfaces'] != 'stage_floor_boxes'
                    or occlusion not in ('obb_nearest_linked_visible', 'none')
                    or occlusion != 'none' and self._cliff['sensor_unknown'] != 2.):
                raise ValueError('Unsupported fan sensor observation contract')
        if os.environ.get('SENSOR_VIS') == '1':
            cfg['env']['enableDebugVis'] = True
        self._sensor_points = self._cliff['sensor_resolution'] ** 2
        self.sensor_metadata = {'version': 3 if self._fan_sensor else 1, 'fusion': 'late_actor_and_critic',
                                'human_height': 'stage_relative',
                                'cliff': copy.deepcopy(self._cliff)}
        if (cfg['env']['numAgents'], cfg['env']['numObjects']) != (2, 3):
            raise ValueError('Cliff scenes require two agents and three objects')
        if cfg['env']['box']['reset']['randomHeight'] or cfg['env']['rootHeightObs']:
            raise ValueError('Cliff requires ground objects and height-invariant AMP observations')
        if cfg['env']['stateInit'] != 'Random':
            raise ValueError('Cliff uses loco reference-state initialization')
        for row in cfg['env']['templateRsi'].values():
            if row != [1.] + [0.] * (len(row) - 1):
                raise ValueError('All cliff task templates must use loco RSI only')
        preset = getattr(cfg['args'], 'task_graph', '')
        presets = ('', 'random_scenario', 'holding', 'sit', 'climb')
        if self._supports_holding_at:
            presets += ('holding_at',)
        if preset not in presets:
            raise ValueError('Unsupported sensor task preset: ' + preset)
        if not preset:
            cfg['args'].task_graph = 'random_scenario'
        if cfg['args'].eval and cfg['env']['eval']['skill'] != ['loco']:
            raise ValueError('Cliff evaluation also requires loco RSI')
        probabilities = cfg['env']['relationGraph']['template_probabilities']
        if (probabilities['HOLDING_AT'] and not self._supports_holding_at) or probabilities['HOLDING_ON_TOP']:
            raise ValueError('Placement tasks are unsupported for this sensor scene')
        super().__init__(cfg, *args, **kwargs)
        if self._square_sensor:
            self._sensor_grid = sensor_grid(self._cliff['sensor_resolution'],
                                            self._cliff['sensor_range'], self.device)
        elif self._fan_sensor:
            self._sensor_grid, self._fan_successors = uniform_fan_grid(
                self._cliff['sensor_resolution'], self._cliff['sensor_range'],
                self._cliff['sensor_fov'], self.device)
            self._sensor_head_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.humanoid_handles_all[0][0], 'head')
            if self._sensor_head_id < 0:
                raise ValueError('Fan sensor requires a head rigid body')
        else:
            self._sensor_grid = sensor_grid(self._cliff['sensor_resolution'],
                                            self._cliff['sensor_extent'], self.device)
        self._cliff_metrics = torch.zeros(4 if self._supports_holding_at else 3, 8, device=self.device)
        self._cliff_agent_counts = torch.zeros(2, 4, device=self.device)
        self._cliff_episode_steps = torch.zeros(2, device=self.device)

    def get_obs_size(self):
        return super().get_obs_size() + self.num_agents * self._sensor_points

    def _create_envs(self, num_envs, spacing, num_per_row):
        low, high = self._cliff['size_range']
        step = self._cliff['size_step']
        max_extent = high / 2 + self._cliff['object_gap'][1] + 1.
        if low < 4 or high < low or step <= 0 or spacing <= max_extent:
            raise ValueError('Invalid cliff dimensions or insufficient envSpacing')
        sizes = np.arange(low, high + step / 2, step)
        indices = np.random.randint(len(sizes), size=(num_envs, 2))
        dimensions = sizes[indices]
        self._stage_half_size = torch.tensor(dimensions / 2, device=self.device, dtype=torch.float)
        options = self._stage_asset_options()
        self._stage_assets = {}
        self._stage_dimensions = dimensions
        self._stage_handles = []
        for width, length in set(map(tuple, dimensions)):
            self._stage_assets[(width, length)] = self.gym.create_box(
                self.sim, float(width), float(length), self._cliff['height'], options)
        super()._create_envs(num_envs, spacing, num_per_row)

    def _stage_asset_options(self):
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        return options

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        pose = gymapi.Transform()
        origin = self._stage_origin(env_id, env_ptr)
        pose.p = gymapi.Vec3(origin.x, origin.y, origin.z + self._cliff['height'] / 2)
        asset = self._stage_assets[tuple(self._stage_dimensions[env_id])]
        handle = self.gym.create_actor(env_ptr, asset, pose, 'cliff_stage', env_id, 0)
        self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(0.32, 0.36, 0.42))
        self._stage_handles.append(handle)

    def _stage_origin(self, env_id, env_ptr):
        return self.gym.get_env_origin(env_ptr)

    def _compute_clean_humanoid_nodes(self, body_pos, body_rot, body_vel, body_ang_vel, origins, env_ids=None):
        local_height_pos = body_pos.clone()
        local_height_pos[..., 2] -= self._cliff['height'] + origins[:, None, None, 2]
        return super()._compute_clean_humanoid_nodes(
            local_height_pos, body_rot, body_vel, body_ang_vel, origins)

    def _compute_clean_scene_obs(self, env_ids=None):
        base = super()._compute_clean_scene_obs(env_ids)
        if self._fan_sensor:
            data = self._fan_sensor_data(env_ids)
            if self.viewer is not None and self.debug_viz:
                self._cache_fan_view(data, env_ids)
            heights = data['observation']
            return torch.cat((base, heights.flatten(1)), -1)
        ids = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            root = self._rigid_body_pos[..., 0, :]
            rotation = self._rigid_body_rot[..., 0, :]
        else:
            body = self._kinematic_humanoid_rigid_body_states[env_ids]
            root, rotation = body[..., 0, :3], body[..., 0, 3:7]
        heading = torch_utils.calc_heading(rotation.reshape(-1, 4)).view(-1, self.num_agents)
        heights = height_observation(root, heading, self._sensor_grid,
            self._env_origins[ids], self._stage_half_size[ids], self._cliff['height'],
            self._cliff['sensor_clip'])
        return torch.cat((base, heights.flatten(1)), -1)

    def _cache_fan_view(self, data, env_ids):
        index = 0
        if env_ids is not None:
            matches = (env_ids == 0).nonzero().flatten()
            if not len(matches):
                return
            index = matches[0].item()
        self._fan_view_data = {key: data[key][index].clone()
                               for key in ('xy', 'filled', 'status', 'eye')}

    def _fan_sensor_data(self, env_ids=None, live=False):
        ids = slice(None) if env_ids is None else env_ids
        if env_ids is None or live:
            pos, rot = self._rigid_body_pos[ids], self._rigid_body_rot[ids]
        else:
            body = self._kinematic_humanoid_rigid_body_states[env_ids]
            pos, rot = body[..., :3], body[..., 3:7]
        if self._square_sensor:
            eye = pos[..., 0, :]
        else:
            head_rot = rot[..., self._sensor_head_id, :].reshape(-1, 4)
            offset = pos.new_tensor(self._cliff['sensor_head_offset']).expand(len(head_rot), -1)
            eye = pos[..., self._sensor_head_id, :] + quat_rotate(head_rot, offset).view(-1, self.num_agents, 3)
        heading = torch_utils.calc_heading(rot[..., 0, :].reshape(-1, 4)).view(-1, self.num_agents)
        centers, quaternions, half_sizes = self._sensor_boxes(ids)
        floor = self._env_origins[ids, 2]
        if self._cliff['sensor_occlusion'] == 'none':
            return fan_height_observation(eye, pos[..., 0, :], heading, self._sensor_grid,
                centers, quaternions, half_sizes, floor, self._cliff['sensor_clip'])
        return fan_observation(eye, pos[..., 0, :], heading, self._sensor_grid,
            centers, quaternions, half_sizes,
            floor, self._cliff['sensor_resolution'], self._cliff['sensor_clip'],
            successors=self._fan_successors, object_batch_size=4 if self.num_envs <= 4 else 1)

    def _sensor_boxes(self, ids):
        origins = self._env_origins[ids]
        stage_center = origins.clone()
        stage_center[:, 2] += self._cliff['height'] / 2
        stage_half = torch.cat((self._stage_half_size[ids],
                               torch.full_like(origins[:, :1], self._cliff['height'] / 2)), -1)
        stage_quat = torch.zeros_like(self._box_states[ids, :1, 3:7])
        stage_quat[..., 3] = 1
        return (torch.cat((stage_center[:, None], self._box_states[ids, :, :3]), 1),
                torch.cat((stage_quat, self._box_states[ids, :, 3:7]), 1),
                torch.cat((stage_half[:, None], self._box_size[ids] / 2), 1))

    def _reset_envs(self, env_ids):
        if not len(env_ids):
            return
        self._sample_episode_graph(env_ids)
        self._reset_default_slots = None
        self._reset_ref_slots = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        self._reset_actors(env_ids)
        self._sample_box_assignments(env_ids)
        logical_sizes = self._logical_box_values(self._box_size, env_ids)
        human_xy, object_xy, object_base = self._sample_scene(env_ids, logical_sizes)
        n = len(env_ids)
        logical = torch.zeros(n, self.num_objects, 13, device=self.device)
        logical[..., :2] = object_xy + self._env_origins[env_ids, None, :2]
        logical[..., 2] = object_base + logical_sizes[..., 2] / 2 + self._env_origins[env_ids, None, 2] + 0.01
        axis = torch.zeros(n * self.num_objects, 3, device=self.device)
        axis[:, 2] = 1
        logical[..., 3:7] = quat_from_angle_axis(
            torch.rand(n * self.num_objects, device=self.device) * (2 * torch.pi), axis).view(n, -1, 4)
        self._box_states[env_ids[:, None], self._logical_box_order[env_ids]] = logical
        slot_env, slot_agent = self._expand_slots(env_ids)
        target_index = agent_object_indices(self.relation_runtime.graph, slot_env, slot_agent).view(n, 2)
        target_xy = logical.gather(1, target_index[..., None].expand(-1, -1, 13))[..., :2]
        human_xy += self._env_origins[env_ids, None, :2]
        direction = target_xy - human_xy
        yaw = torch.atan2(direction[..., 1], direction[..., 0])
        yaw += (torch.rand_like(yaw) * 2 - 1) * self._cliff['heading_jitter']
        self._place_humans(env_ids, human_xy, yaw)
        self._reset_task(env_ids)
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(env_ids)
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)

    def _sample_scene(self, env_ids, logical_sizes):
        human_xy, object_xy = sample_layout(self._stage_half_size[env_ids], logical_sizes,
            self._cliff['spawn_margin'], self._cliff['object_gap'])
        return human_xy, object_xy, torch.zeros_like(object_xy[..., 0])

    def _place_humans(self, ids, xy, yaw):
        root = self._humanoid_root_states[ids].clone()
        body = self._kinematic_humanoid_rigid_body_states[ids].clone()
        old_heading = torch_utils.calc_heading(root[..., 3:7].reshape(-1, 4)).view_as(yaw)
        axis = torch.zeros(yaw.numel(), 3, device=self.device)
        axis[:, 2] = 1
        q = quat_from_angle_axis((yaw - old_heading).flatten(), axis).view(len(ids), 2, 4)
        qb = q[:, :, None].expand(-1, -1, self.num_bodies, -1).reshape(-1, 4)
        offset = body[..., :3] - root[..., None, :3]
        root[..., :2] = xy
        root[..., 2] += self._cliff['height']
        root[..., 3:7] = quat_mul(q.reshape(-1, 4), root[..., 3:7].reshape(-1, 4)).view(len(ids), 2, 4)
        for start in (7, 10):
            root[..., start:start + 3] = quat_rotate(q.reshape(-1, 4),
                root[..., start:start + 3].reshape(-1, 3)).view(len(ids), 2, 3)
            body[..., start:start + 3] = quat_rotate(qb,
                body[..., start:start + 3].reshape(-1, 3)).view(len(ids), 2, self.num_bodies, 3)
        body[..., :3] = quat_rotate(qb, offset.reshape(-1, 3)).view_as(offset) + root[..., None, :3]
        body[..., 3:7] = quat_mul(qb, body[..., 3:7].reshape(-1, 4)).view(len(ids), 2, self.num_bodies, 4)
        self._humanoid_root_states[ids] = root
        self._kinematic_humanoid_rigid_body_states[ids] = body

    def _compute_reset(self):
        ground = self._terrain_height(self._rigid_body_pos[..., :2])
        fallen = terrain_fall(self._contact_forces, self._rigid_body_pos,
            self._contact_body_ids, ground, self._termination_heights, self.progress_buf,
            self._env_origins[:, 2])
        if not self._enable_early_termination:
            fallen.zero_()
        self._terminate_buf[:] = fallen.flatten().long()
        self.reset_buf[:] = (fallen.any(-1) | (self.progress_buf >= self.max_episode_length - 1)).long()
        self._record_cliff_metrics(fallen)

    def _terrain_height(self, xy):
        return stage_height(xy, self._env_origins, self._stage_half_size, self._cliff['height'])

    def _record_cliff_metrics(self, fallen):
        root = self._rigid_body_pos[..., 0, :]
        xy = root[..., :2] - self._env_origins[:, None, :2]
        clearance = (self._stage_half_size[:, None] - xy.abs()).amin(-1)
        rotation = self._rigid_body_rot[..., 0, :]
        up_z = 1 - 2 * (rotation[..., 0].square() + rotation[..., 1].square())
        upright = (up_z > .7) & (root[..., 2] - self._env_origins[:, None, 2] > self._cliff['height'] + .6)
        on_stage = (clearance >= 0) & upright
        near = on_stage & (clearance < 1.)
        speed = self._rigid_body_vel[..., 0, :2].norm(dim=-1)
        stopped = near & (speed < 0.15)
        turning = near & (self._rigid_body_ang_vel[..., 0, 2].abs() > 0.3)
        slot_env, slot_agent = self._expand_slots(torch.arange(self.num_envs, device=self.device))
        kinds = classify_templates(self.relation_runtime.graph, slot_env, slot_agent, True).view(-1, 2)
        truncated = fallen.any(-1)[:, None] & ~fallen
        active = getattr(self, '_cliff_diagnostic_active', torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        sensor = self.obs_buf[:, -2 * self._sensor_points:].reshape(-1, 2, self._sensor_points)
        sees_edge = sensor.amax(-1) - sensor.amin(-1) > .1
        if self._fan_sensor:
            known = sensor != 2
            highest = sensor.masked_fill(~known, -torch.inf).amax(-1)
            lowest = sensor.masked_fill(~known, torch.inf).amin(-1)
            sees_edge = (known.sum(-1) >= 2) & (highest - lowest > .1)
        for k in range(len(self._cliff_metrics)):
            mask = (kinds == k) & active[:, None]
            self._cliff_metrics[k] += torch.stack((mask.sum(), (near & mask).sum(),
                (stopped & mask).sum(), (turning & mask).sum(), (fallen & mask).sum(),
                (truncated & mask).sum(), (sees_edge & mask).sum(), (speed * near * mask).sum()))
        self._cliff_agent_counts += torch.stack(tuple(
            (value & active[:, None]).sum(0) for value in (fallen, truncated, near, sees_edge)), -1)
        endings = self.reset_buf * active
        self._cliff_episode_steps += torch.stack(((self.progress_buf * endings).sum(), endings.sum()))

    def cliff_diagnostics(self, reset=False):
        result = {}
        for name, row in zip(('holding', 'sit', 'climb', 'holding_at'), self._cliff_metrics):
            for key, value in zip(('agent_steps', 'near_edge_steps', 'stopped_near_edge_steps',
                                  'turning_near_edge_steps', 'falls', 'teammate_truncated',
                                  'sensor_edge_steps'), row[:7]):
                result[name + '/' + key] = value.item()
            result[name + '/near_edge_speed'] = (row[7] / row[1].clamp_min(1)).item()
        for agent, row in enumerate(self._cliff_agent_counts):
            for key, value in zip(('falls', 'teammate_truncated', 'near_edge_steps', 'sensor_edge_steps'), row):
                result['agent_{}/{}'.format(agent, key)] = value.item()
        result['episode_length'] = (self._cliff_episode_steps[0] / self._cliff_episode_steps[1].clamp_min(1)).item()
        if reset:
            self._cliff_metrics.zero_()
            self._cliff_agent_counts.zero_()
            self._cliff_episode_steps.zero_()
        return result

    def _draw_ontop_context(self):
        super()._draw_ontop_context()
        if not self.debug_viz:
            return
        if self._fan_sensor:
            self._draw_fan_sensor()
            return
        root = self._rigid_body_pos[:1, :, 0]
        heading = torch_utils.calc_heading(self._rigid_body_rot[:1, :, 0].reshape(-1, 4)).view(1, 2)
        xy = sensor_positions(root, heading, self._sensor_grid)
        z = stage_height(xy, self._env_origins[:1], self._stage_half_size[:1], self._cliff['height'])
        p = torch.cat((xy, z[..., None] + 0.03), -1)[0] - self._env_origins[0]
        lines = torch.stack((p, p + p.new_tensor([0, 0, 0.05])), -2).cpu().numpy()
        for agent, color in enumerate(((0.2, 0.8, 1.), (1., 0.6, 0.2))):
            colors = np.tile(np.array(color, dtype=np.float32), (self._sensor_points, 1))
            self.gym.add_lines(self.viewer, self.envs[0], self._sensor_points, lines[agent], colors)

    def _draw_fan_sensor(self):
        from env.tasks.multi_agent.vision_sensor_debug import draw_fan_sensor
        if self._fan_view_data is None:
            return
        origin = self.gym.get_env_origin(self.envs[0])
        draw_fan_sensor(self.gym, self.viewer, self.envs[0],
                        self._fan_view_data, self._env_origins.new_tensor([origin.x, origin.y, origin.z]),
                        self._cliff['sensor_resolution'], uniform_rings=True, square=self._square_sensor)
