import torch

from env.tasks.multi_agent.compact_geometry import CompactGeometry
from env.tasks.multi_agent.compact_rsi import CompactRsi
from env.tasks.multi_agent.humanoid_ma_sensor_cliff_bridge import HumanoidMASensorCliffBridge
from utils.compact_bridge import KINDS, choose_boundary, sample_distance
from utils.cliff_bridge_spec import COOPERATIVE, CONTROLLED, FREE
from env.tasks.multi_agent import square_sensor_scene as scene


class HumanoidMASensorCompact(CompactGeometry, CompactRsi, HumanoidMASensorCliffBridge):
    def _init_bridge_curriculum(self):
        super()._init_bridge_curriculum()
        self._compact_distance = torch.zeros(self.num_envs, device=self.device)
        self._compact_kind = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self._compact_requested = torch.zeros_like(self._compact_kind)
        self._compact_stats = torch.zeros(4, 3, device=self.device)
        self._compact_cross_stats = torch.zeros(4, 8, device=self.device)
        self._compact_reset_stats = torch.zeros(5, device=self.device)
        self._compact_failures = torch.zeros(3, device=self.device)
        self._compact_platform_counts = torch.zeros(2,4,device=self.device)
        self._compact_depth_stats = torch.zeros(2,device=self.device)

    def _curriculum_surfaces(self, ids):
        super()._curriculum_surfaces(ids)
        self._cross_phase[ids] = torch.where(self._crosser_mask[ids], 0, -1)
        self._cross_progress[ids] = 0.
        branch = self._bridge_branch[ids]
        connected = (branch == FREE) | ((branch == CONTROLLED) & self._controlled_present[ids])
        connected |= (branch == COOPERATIVE) & (self._bridge_start_mode[ids] == 0)
        eligible = self._crosser_mask[ids] & connected[:,None]
        eligible &= ~((branch == COOPERATIVE) & self._bridge_carry_from_start[ids])[:,None]
        kind = choose_boundary(eligible, self._scene_cfg['bridge']['boundary_rsi'], self.cfg['args'].eval)
        forced = getattr(self, '_compact_test_kind', None)
        if forced is not None:
            if not self.cfg['args'].eval:
                raise ValueError('Forced boundary resets are evaluation-only')
            kind.zero_()
            rows = eligible.any(-1).nonzero().flatten()
            person = eligible.long().argmax(-1)
            kind[rows,person[rows]] = forced
        self._compact_kind[ids], self._compact_requested[ids] = kind, kind

    def _sample_episode_graph(self, ids):
        super()._sample_episode_graph(ids)
        spec = self._scene_cfg['bridge']['distance_curriculum']
        self._compact_distance[ids] = sample_distance(len(ids), self._bridge_curriculum_update,
            spec, self.device, self.cfg['args'].eval)
        forced = getattr(self, '_compact_test_alpha', None)
        if forced is not None:
            if not self.cfg['args'].eval:
                raise ValueError('Forced distances are evaluation-only')
            self._compact_distance[ids] = forced
        self._activate_compact_geometry(ids)

    def _place_scene(self, ids):
        self._cross_phase[ids] = torch.where(self._crosser_mask[ids],0,-1)
        self._cross_progress[ids] = 0.
        super()._place_scene(ids)
        roots = self._humanoid_root_states[ids].clone()
        bodies = self._kinematic_humanoid_rigid_body_states[ids].clone()
        goals = self._tar_pos[ids].clone()
        self._apply_boundary_rsi(ids)
        from env.tasks.multi_agent.stage2_layout import invalid_before_layout
        bad = invalid_before_layout(self,ids) & ~self._place_failed[ids]
        rejected = ids[bad]
        self._compact_failures[2] += (self._compact_kind[rejected] > 0).sum()
        self._humanoid_root_states[rejected] = roots[bad]
        self._kinematic_humanoid_rigid_body_states[rejected] = bodies[bad]
        self._tar_pos[rejected] = goals[bad]
        self._compact_kind[rejected] = 0
        self._cross_phase[rejected] = torch.where(self._crosser_mask[rejected],0,-1)
        self._cross_progress[rejected] = 0.

    def _reset_env_tensors(self, ids):
        for i in range(4):
            self._compact_stats[i,0] += (self._compact_requested[ids] == i).logical_and(self._crosser_mask[ids]).sum()
            self._compact_stats[i,1] += (self._compact_kind[ids] == i).logical_and(self._crosser_mask[ids]).sum()
        self._compact_stats[:,2] += torch.bincount(self._compact_requested[ids][
            self._compact_requested[ids] != self._compact_kind[ids]], minlength=4)
        bridge = self._scene_ids[ids] == 1
        for side in (0,1):
            self._compact_reset_stats[side] += self._compact_sizes[ids[bridge],side].sum()
        self._compact_reset_stats[2] += 2*bridge.sum()
        self._compact_reset_stats[3] += self._compact_distance[ids[bridge]].sum()
        self._compact_reset_stats[4] += bridge.sum()
        es = ids[bridge]
        for side in (0,1):
            self._compact_platform_counts[side] += torch.bincount(
                (self._compact_visitors[es,side]-1).clamp(0,3),minlength=4)
        if len(es):
            local = scene.rotate_xy(self._humanoid_root_states[es,:,:2]-self._env_origins[es,None,:2]-
                self._structure_shift[es,1,None],-self._structure_yaw[es,1,None])
            index = (self._start_surface[es]-1).clamp_min(0)
            center = self._surface_center[es[:,None],1,index,0]
            sign = torch.sign(center-self._bridge_landing[es,0,None])
            depth = (local[...,0]-self._bridge_landing[es,0,None])*sign-self._bridge_gap[es,None]/2
            full = self._crosser_mask[es] & (self._compact_kind[es] == 0)
            self._compact_depth_stats[0] += depth[full].sum()
            self._compact_depth_stats[1] += full.sum()
        super()._reset_env_tensors(ids)

    def _record_variant_crossing(self, fallen, ending, roots):
        super()._record_variant_crossing(fallen, ending, roots)
        eligible = self._crosser_mask & ending[:,None]
        other = fallen.any(-1)[:,None] & ~fallen
        entered = self._cross_entered & (self._compact_kind < 2)
        values = torch.stack((torch.ones_like(eligible), entered, self._cross_completed,
            fallen, other, self._bridge_arrived[:,None].expand_as(eligible),
            (self.progress_buf >= self.max_episode_length)[:,None].expand_as(eligible),
            other & ~self._cross_completed),-1).float()
        self._compact_cross_stats.index_add_(0, self._compact_kind[eligible], values[eligible])

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        for i, name in enumerate(KINDS):
            for j, metric in enumerate(('requested','executed','fallback')):
                result['compact/reset/'+name+'/'+metric] = self._compact_stats[i,j].clone()
            self._crossing_metrics(result, 'compact/crossing/'+name+'/', self._compact_cross_stats[i])
        s = self._compact_reset_stats
        result['compact/platform/mean_side_a'] = s[0]/s[2].clamp_min(1)
        result['compact/platform/mean_side_b'] = s[1]/s[2].clamp_min(1)
        result['compact/distance/mean_alpha'] = s[3]/s[4].clamp_min(1)
        for i, reason in enumerate(('motion','support_or_collision','before_layout')):
            result['compact/reset/fallback_'+reason] = self._compact_failures[i].clone()
        for side,name in enumerate(('a','b')):
            for i in range(4):
                result[f'compact/platform/{name}/people{i+1}'] = self._compact_platform_counts[side,i].clone()
        d = self._compact_depth_stats
        result['compact/distance/source_depth_mean'] = d[0]/d[1].clamp_min(1)
        self._compact_stats.zero_()
        self._compact_cross_stats.zero_()
        self._compact_reset_stats.zero_()
        self._compact_failures.zero_()
        self._compact_platform_counts.zero_()
        self._compact_depth_stats.zero_()
        return result
