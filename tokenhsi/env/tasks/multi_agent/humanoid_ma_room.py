import copy
import numpy as np
from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.bridge_geometry import compute_agent_cpa_collision_penalty
from env.tasks.multi_agent.room_geometry import (
    compute_wall_cpa_collision_penalty, room_walls, sample_room_layout, sample_partition_layout)
from utils.edge_scenario_spec import agent_object_indices, agent_goal_indices, classify_templates


class HumanoidMARoom(HumanoidMACliff):
    def __init__(self, cfg, *args, **kwargs):
        self._room = cfg['env']['room']
        self._supports_holding_at = 'partition' in self._room
        self._passage_width = self._room['partition']['passage_width'] if self._supports_holding_at else None
        self._agent_collision_ttc_discount = cfg['env']['agentCollisionTTCDiscount']
        if not 0 < self._agent_collision_ttc_discount <= 1:
            raise ValueError('agentCollisionTTCDiscount must be in (0, 1]')
        self._wall_collision_ttc_discount = cfg['env']['wallCollisionTTCDiscount']
        if not 0 < self._wall_collision_ttc_discount <= 1:
            raise ValueError('wallCollisionTTCDiscount must be in (0, 1]')
        if cfg['env']['cliff']['sensor_occlusion'] != 'none':
            raise ValueError('Room uses the height sensor without occlusion')
        super().__init__(cfg, *args, **kwargs)
        self.sensor_metadata.update(version=6 if self._supports_holding_at else 5, room=copy.deepcopy(self._room))
        self._wall_centers += self._env_origins[:, None]
        self._wall_starts += self._env_origins[:, None, :2]
        self._wall_ends += self._env_origins[:, None, :2]
        if self._supports_holding_at:
            self._room_goal_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if env_id == 0:
            self._wall_centers, self._wall_half_size, self._wall_starts, self._wall_ends = room_walls(
                self._stage_half_size, self._room['wall_thickness'],
                self._cliff['height'] + self._room['wall_height'], self._passage_width)
            self._wall_assets = {}
            self._wall_handles = []
        origin = self.gym.get_env_origin(env_ptr)
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        handles = []
        for side in range(self._wall_centers.shape[1]):
            size = tuple((2 * self._wall_half_size[env_id, side]).cpu().tolist())
            if size not in self._wall_assets:
                self._wall_assets[size] = self.gym.create_box(self.sim, *size, options)
            pose = gymapi.Transform()
            center = self._wall_centers[env_id, side].cpu().tolist()
            pose.p = gymapi.Vec3(origin.x + center[0], origin.y + center[1], origin.z + center[2])
            handle = self.gym.create_actor(env_ptr, self._wall_assets[size], pose, 'room_wall_' + str(side), env_id, 0)
            color = ((0.52, 0.57, 0.64) if side == 4 else
                     (0.64, 0.69, 0.76) if side < 2 else (0.74, 0.70, 0.62))
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            handles.append(handle)
        self._wall_handles.append(handles)

    def _sample_scene(self, env_ids, logical_sizes):
        if self._supports_holding_at:
            env, agent = self._expand_slots(env_ids)
            assigned = agent_object_indices(self.relation_runtime.graph, env, agent).view(-1, 2)
            kinds = classify_templates(self.relation_runtime.graph, env, agent, True).view(-1, 2)
            pickup_cross = ((kinds == 0) | (kinds == 3)) & (torch.rand(kinds.shape, device=self.device)
                < self._room['partition'].get('pickup_cross_probability', 0.))
            human_xy, object_xy, goals = sample_partition_layout(self._stage_half_size[env_ids], logical_sizes,
                assigned, self._cliff['spawn_margin'], self._room['object_margin'],
                self._room['wall_thickness'], self._passage_width, pickup_cross)
            self._room_goal_xy[env_ids] = goals
        else:
            human_xy, object_xy = sample_room_layout(self._stage_half_size[env_ids], logical_sizes,
                self._cliff['spawn_margin'], self._room['object_margin'])
        return human_xy, object_xy, torch.full_like(object_xy[..., 0], self._cliff['height'])

    def _reset_task(self, env_ids):
        self._tar_pos[env_ids] = self._env_origins[env_ids, None]
        self._tar_pos[env_ids, :, 2] += self._cliff['height']
        if not self._supports_holding_at:
            return
        self._tar_pos[env_ids, :, 1] += (self._stage_half_size[env_ids, 1] - self._passage_width / 2)[:, None]
        env, agent = self._expand_slots(env_ids)
        active = classify_templates(self.relation_runtime.graph, env, agent, True) == 3
        env, agent = env[active], agent[active]
        goals = agent_goal_indices(self.relation_runtime.graph, env, agent)
        physical = self._scenario_physical_object(env, agent)
        self._tar_pos[env, goals, :2] = self._room_goal_xy[env, agent] + self._env_origins[env, :2]
        self._tar_pos[env, goals, 2] = self._env_origins[env, 2] + self._cliff['height'] + self._box_size[env, physical, 2] / 2
        if self._enable_markers:
            for e, a, g in zip(env.tolist(), agent.tolist(), goals.tolist()):
                handle = self._marker_handles[e * self.num_agents + g]
                self.gym.set_rigid_body_color(self.envs[e], handle, 0, gymapi.MESH_VISUAL, self._agent_color(a))

    def _sensor_boxes(self, ids):
        centers, quaternions, half_sizes = super()._sensor_boxes(ids)
        wall_quats = torch.zeros_like(self._wall_centers[ids, :, :1]).expand(-1, -1, 4).clone()
        wall_quats[..., 3] = 1
        return (torch.cat((centers, self._wall_centers[ids]), 1),
                torch.cat((quaternions, wall_quats), 1),
                torch.cat((half_sizes, self._wall_half_size[ids]), 1))

    def _terrain_height(self, xy):
        # Feet touching a wall still stand on the room floor; wall tops are not support terrain.
        return (self._env_origins[:, 2] + self._cliff['height']).view(
            (-1,) + (1,) * (xy.ndim - 2)).expand(xy.shape[:-1])

    def _compute_reward(self, actions):
        return self._compute_relation_reward(self._cpa_collision_risk)

    def _cpa_collision_risk(self, roots, min_dist):
        velocity = self._humanoid_root_states[..., 7:10]
        agents = compute_agent_cpa_collision_penalty(
            roots, velocity, min_dist, self.dt, self._agent_collision_ttc_discount)
        walls = compute_wall_cpa_collision_penalty(roots, velocity, self._wall_starts,
            self._wall_ends, min_dist, self.dt, self._wall_collision_ttc_discount)
        return torch.maximum(agents, walls)

    def _draw_ontop_context(self):
        super()._draw_ontop_context()
        if self._supports_holding_at:
            self._draw_room_goals()

    def _draw_room_goals(self):
        env = torch.zeros(2, dtype=torch.long, device=self.device)
        agents = torch.arange(2, device=self.device)
        active = classify_templates(self.relation_runtime.graph, env, agents, True) == 3
        goals = agent_goal_indices(self.relation_runtime.graph, env, agents)
        angle = np.linspace(0, 2 * np.pi, 33)
        for agent in agents[active].tolist():
            goal = (self._tar_pos[0, goals[agent]] - self._env_origins[0]).cpu().numpy()
            points = np.stack((goal[0] + .3 * np.cos(angle), goal[1] + .3 * np.sin(angle),
                               np.full_like(angle, self._cliff['height'] + .025)), -1)
            lines = np.stack((points[:-1], points[1:]), 1).astype(np.float32)
            color = self._agent_color(agent)
            colors = np.tile([color.x, color.y, color.z], (32, 1)).astype(np.float32)
            self.gym.add_lines(self.viewer, self.envs[0], len(lines), lines, colors)
