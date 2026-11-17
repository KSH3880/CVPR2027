import os

import torch

from env.tasks.multi_agent.humanoid_ma_sensor_compact import HumanoidMASensorCompact
from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent import square_sensor_scene as scene
from utils.bridge_before_fixed import FOLLOWER_TASKS, PUSHERS, fixed_roles, bridge_center_progress
from utils.sensor_stage2_spec import SURFACE_A

STATS = ('entry_left', 'entry_right', 'follower_sit', 'follower_climb', 'follower_ontop')


class HumanoidMASensorBridgeBefore(HumanoidMASensorCompact):
    """Bridge with two pushers and a BEFORE pair on the source platform or across the bridge."""

    def __init__(self, cfg, *args, **kwargs):
        if os.environ.get('SCENARIO', 'bridge') != 'bridge':
            raise ValueError('Fixed BEFORE Bridge supports only SCENARIO=bridge')
        if os.environ.get('BRIDGE_MODE', 'cooperative') != 'cooperative':
            raise ValueError('Fixed BEFORE Bridge supports only BRIDGE_MODE=cooperative')
        if os.environ.get('BRIDGE_PUSHERS', str(PUSHERS)) != str(PUSHERS):
            raise ValueError('Fixed BEFORE Bridge always uses two pushers')
        super().__init__(cfg, *args, **kwargs)

    def _init_bridge_curriculum(self):
        super()._init_bridge_curriculum()
        self._fixed_before_stats = torch.zeros(len(STATS), device=self.device)
        self._before_same_platform = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bridge_center_passed = torch.zeros_like(self._before_same_platform)
        self._bridge_center_direction = torch.zeros(self.num_envs, 2, device=self.device)
        self._before_mix_counts = torch.zeros(2, device=self.device)

    def _sample_bridge_curriculum(self, ids, counts):
        return super()._sample_bridge_curriculum(ids, torch.full_like(counts, PUSHERS))

    def _draw_episode_graph(self, ids):
        domain, groups, pre, tasks = super()._draw_episode_graph(ids)
        roles = self._roles[ids]
        crosser, follower = roles[:, 1], roles[:, 3]
        groups, pre, tasks = fixed_roles(tasks, groups, roles, self._box_size[ids, self._leader_bank+crosser],
            self.cfg['env']['box']['build']['taskSizeRanges'])
        training = self._scene_cfg['bridge'].get('before_training')
        same = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if training is not None:
            same = torch.rand(len(ids), device=self.device) < training['same_platform_probability']
        self._set_before_surfaces(ids, same)
        self._bridge_carry_from_start[ids] = True
        # With side=True the bridge enters from +y, i.e. from the left of a crosser starting on platform A.
        left = self._bridge_side[ids] == (self._start_surface[ids, crosser] == SURFACE_A)
        family = tasks[torch.arange(len(ids), device=self.device), follower]
        counts = [left.sum(), (~left).sum()]+[(family == task).sum() for task in FOLLOWER_TASKS]
        self._fixed_before_stats += torch.stack(counts).float()
        return domain, groups, pre, tasks

    def _set_before_surfaces(self, ids, same):
        crosser, follower = self._roles[ids, 1], self._roles[ids, 3]
        start = self._start_surface[ids, crosser]
        dest = torch.where(same, start, 3-start)
        self._before_same_platform[ids] = same
        self._surface[ids, crosser] = dest
        self._surface[ids, follower] = dest
        self._start_surface[ids, follower] = dest
        self._curriculum_surfaces(ids)
        self._crosser_mask[ids[same]] = False

    def _place_bridge(self, ids, rows, occupied):
        # The generic scene loop skips the bridge leader, so place its BEFORE group here.
        same = self._before_same_platform[ids[rows]]
        crossing = rows[~same]
        if len(crossing):
            super()._place_bridge(ids, crossing, occupied)
        if same.any():
            SensorStage2Placement._place_bridge(self, ids, rows[same], occupied, place_crossers=False)
        entities = self._group_entities(ids)
        head = torch.arange(self.num_agents, device=self.device).expand(len(ids), -1)
        before_head = torch.where(self._before_predecessor[ids] >= 0, self._before_predecessor[ids], head)
        follower = self._roles[ids[crossing], 3]
        for person in range(self.num_agents):
            selected = crossing[follower == person]
            if len(selected):
                self._place_group(ids, selected, person, head, entities, occupied)
            local = rows[same]
            selected = local[self._roles[ids[local], 1] == person]
            if len(selected):
                self._place_group(ids, selected, person, before_head, entities, occupied)

    def _place_scene(self, ids):
        super()._place_scene(ids)
        if self._state_init == self.StateInit.Default:
            follower = self._roles[ids, 3]
            self._tar_pos[ids, follower] = self._humanoid_root_states[ids, follower, :3]

    def _reset_relation_history(self, ids):
        super()._reset_relation_history(ids)
        self._bridge_center_passed[ids] = False
        crosser = self._roles[ids, 1]
        start = self._start_surface[ids, crosser]-1
        centers = self._surface_center[ids, 1]
        rows = torch.arange(len(ids), device=self.device)
        direction = centers[rows, 1-start]-centers[rows, start]
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._bridge_center_direction[ids] = scene.rotate_xy(direction, self._structure_yaw[ids, 1])
        same = self._before_same_platform[ids]
        self._before_mix_counts += torch.stack((same.sum(), (~same).sum()))

    def _compute_bridge_center_progress(self):
        spec = self._scene_cfg['bridge']['before_training']['bridge_center_progress']
        ids = torch.arange(self.num_envs, device=self.device)
        crosser = self._roles[:, 1]
        payload = self._agent_box_assignment[ids, crosser]
        value, passed = bridge_center_progress(self._box_states[ids, payload, :2],
            self._box_states[ids, self._bridge_slot, :2], self._bridge_center_direction,
            self._bridge_center_passed, ~self._before_same_platform, spec['sigma'])
        self._bridge_center_passed.copy_(passed)
        reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        reward[ids, crosser] = spec['weight']*value
        return reward

    def _init_amp_obs(self, env_ids):
        if self._state_init == self.StateInit.Default:
            self._reset_default_slots = self._expand_slots(env_ids)
        return super()._init_amp_obs(env_ids)

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        for i, name in enumerate(STATS):
            result['bridge_before/'+name] = self._fixed_before_stats[i].clone()
        self._fixed_before_stats.zero_()
        result['bridge_before/same_platform_episodes'] = self._before_mix_counts[0].clone()
        result['bridge_before/crossing_episodes'] = self._before_mix_counts[1].clone()
        result['bridge_before/same_platform_fraction'] = self._before_mix_counts[0]/self._before_mix_counts.sum().clamp_min(1)
        self._before_mix_counts.zero_()
        return result
