from dataclasses import fields
import time

import torch

from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before import HumanoidMASensorBridgeBefore
from utils.compact_bridge import sample_distance
from utils.episode_layout_cache import EpisodeLayoutCache, buckets, validate


STATE_FIELDS = (
    '_kinematic_humanoid_rigid_body_states', '_every_env_init_dof_pos', '_amp_obs_buf',
    '_tar_pos', '_center_tar_pos', '_env_origins', '_scene_ids', '_scene_origins',
    '_static_centers', '_static_half', '_static_local', '_static_quat',
    '_surface_center', '_surface_half', '_surface_height', '_structure_bounds',
    '_structure_yaw', '_structure_shift', '_structure_axis', '_structure_direction',
    '_structure_near', '_bridge_gap', '_bridge_far', '_bridge_half_a', '_bridge_half_b',
    '_roles', '_surface', '_start_surface', '_unreachable', '_place_failed',
    '_bridge_near', '_bridge_aligned', '_bridge_carry_from_start', '_bridge_side',
    '_bridge_landing', '_bridge_landing_world', '_bridge_start_mode', '_bridge_count',
    '_bridge_slot', '_bridge_pushers', '_bridge_span', '_bridge_walk', '_bridge_branch',
    '_compact_active', '_compact_visitors', '_compact_sizes', '_compact_distance',
    '_compact_kind', '_compact_requested', '_crosser_mask', '_cross_phase', '_cross_progress',
    '_groups', '_templates', '_before_predecessor', '_before_near_pairs', '_members',
    '_regions', '_agent_box_assignment', '_logical_box_order', '_stage2_task_slots',
    '_scene_domains', '_controlled_present', '_controlled_arrival',
    '_before_same_platform',
)


class HumanoidMASensorBridgeLayoutCache(HumanoidMASensorBridgeBefore):
    def __init__(self, cfg, *args, **kwargs):
        self._layout_spec = validate(cfg['env'])
        self._layout_final_only = cfg['env']['sensorScene']['bridge']['distance_curriculum']['mixture'] == [0.,0.,1.]
        self._layout_cache = None
        self._layout_force = None
        self._layout_building = False
        super().__init__(cfg, *args, **kwargs)

    def _layout_tensors(self):
        result = {name: getattr(self, name) for name in STATE_FIELDS}
        result['actors'] = self._root_states.view(self.num_envs, -1, 13)
        result['dofs'] = self._dof_state.view(self.num_envs, -1, 2)
        graph = self.relation_runtime.graph
        result.update({'graph/'+f.name: getattr(graph, f.name) for f in fields(graph)
                       if isinstance(getattr(graph, f.name), torch.Tensor)})
        return result

    def _draw_episode_graph(self, ids):
        domain, groups, pre, tasks = super()._draw_episode_graph(ids)
        if self._layout_force is not None:
            mode, family, _, same = self._layout_force
            self._bridge_start_mode[ids] = mode
            self._bridge_near[ids] = mode == 1
            self._bridge_aligned[ids] = (mode == 0) | ((mode == 1) & (torch.rand(len(ids), device=self.device) < .5))
            rows = torch.arange(len(ids), device=self.device)
            old = tasks[rows, self._roles[ids, 3]]
            for index, task in enumerate((1, 2, 4)):
                self._fixed_before_stats[2+index] += (family == task).sum()-(old == task).sum()
            tasks[rows, self._roles[ids, 3]] = family
            self._set_before_surfaces(ids, same)
        return domain, groups, pre, tasks

    def _sample_episode_graph(self, ids):
        super()._sample_episode_graph(ids)
        if self._layout_force is not None:
            self._compact_distance[ids] = self._layout_force[2]

    def _generate_layouts(self, ids, mode, family, alpha, same=None):
        same = torch.zeros_like(ids, dtype=torch.bool) if same is None else same
        self._layout_force = (mode, family, alpha, same)
        try:
            super()._reset_envs(ids)
        finally:
            self._layout_force = None

    def _prepare_layout_cache(self):
        mixed = self._scene_cfg['bridge'].get('before_training', {}).get('same_platform_probability', 0) > 0
        layouts = 2 if mixed else 1
        distances = (1.,) if self._layout_final_only else (0., 1.)
        self._layout_cache = EpisodeLayoutCache(self.num_envs, self._layout_spec['entries_per_bucket'],
                                                (9 if self._layout_final_only else 27)*layouts)
        self._layout_hits = torch.zeros(2, device=self.device)
        saved = {name: value.clone() for name, value in vars(self).items()
                 if isinstance(value, torch.Tensor) and (
                     name.endswith(('_stats', '_counts', '_samples', '_failures', '_outcomes', '_exposure'))
                     or name.startswith(('_sampling_', '_size_rsi_')) or name in ('_relation_episode_id', '_scene_sampling'))}
        ids = torch.arange(self.num_envs, device=self.device)
        start = time.perf_counter()
        count = 9*layouts*len(distances)*self._layout_spec['entries_per_bucket']
        print(f'[layout-cache] preparing {count} valid layouts per environment', flush=True)
        self._layout_building = True
        try:
            for same in range(layouts):
                for mode in range(3):
                    for family in (1, 2, 4):
                        for distance in distances:
                            for _ in range(self._layout_spec['entries_per_bucket']):
                                modes = torch.full_like(ids, mode)
                                families = torch.full_like(ids, family)
                                alpha = torch.full((len(ids),), distance, device=self.device)
                                local = torch.full_like(ids, bool(same), dtype=torch.bool)
                                self._generate_layouts(ids, modes, families, alpha, local)
                                bucket = buckets(modes, families, alpha, local, self._layout_final_only)
                                self._layout_cache.store(bucket, ids, alpha, self._layout_tensors())
        finally:
            self._layout_building = False
            for name, value in saved.items():
                getattr(self, name).copy_(value)
        print(f'[layout-cache] prepared {count} layouts per env in {time.perf_counter()-start:.1f}s', flush=True)

    def _reset_envs(self, ids):
        if self.cfg['args'].eval or not len(ids) or self._layout_building:
            return super()._reset_envs(ids)
        if self._layout_cache is None:
            self._prepare_layout_cache()
        spec = self._scene_cfg['bridge']
        mode = torch.multinomial(torch.tensor(spec['start_distribution']['probabilities'],
                                             device=self.device), len(ids), True)
        family = torch.tensor((1, 2, 4), device=self.device)[torch.randint(3, (len(ids),), device=self.device)]
        alpha = (torch.ones(len(ids), device=self.device) if self._layout_final_only else
                 sample_distance(len(ids), self._bridge_curriculum_update, spec['distance_curriculum'], self.device))
        probability = spec.get('before_training', {}).get('same_platform_probability', 0)
        same = torch.rand(len(ids), device=self.device) < probability
        kinds = buckets(mode, family, alpha, same, self._layout_final_only)
        tensors = self._layout_tensors()
        if self._layout_final_only and self._layout_spec['refresh_probability'] == 0:
            slots = torch.randint(self._layout_spec['entries_per_bucket'], (len(ids),), device=self.device)
            self._layout_cache.restore(kinds, ids, slots, tensors)
            self._restore_layouts(ids)
            self._layout_hits[0] += len(ids)
            return
        slots = self._layout_cache.lookup(kinds, ids, alpha)
        fresh = (slots < 0) | (torch.rand(len(ids), device=self.device) < self._layout_spec['refresh_probability'])
        if (~fresh).any():
            self._layout_cache.restore(kinds[~fresh], ids[~fresh], slots[~fresh], tensors)
            self._restore_layouts(ids[~fresh])
        if fresh.any():
            self._generate_layouts(ids[fresh], mode[fresh], family[fresh], alpha[fresh], same[fresh])
            self._layout_cache.store(kinds[fresh], ids[fresh], alpha[fresh], tensors)
        self._layout_hits += torch.stack(((~fresh).sum(), fresh.sum()))

    def _restore_layouts(self, ids):
        self._reset_default_slots = None
        self._reset_ref_slots, self._reset_ref_motion_ids, self._reset_ref_motion_times = {}, {}, {}
        for name in ('_bridge_arrived', '_bridge_crossed', '_bridge_entered', '_cross_entered',
                     '_cross_completed', '_cliff_exited', '_cliff_dropped', '_push_hits', '_ontop_maintain_steps'):
            getattr(self, name)[ids] = 0
        self._edge_goal_owners[ids] = True
        runtime = self.relation_runtime
        from utils.stage2_spec import before_members
        runtime.before[ids] = before_members(self._before_predecessor[ids])
        if runtime.relaxed_at_owners is not None:
            runtime.relaxed_at_owners[ids] = self._members[ids] == 1
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        self._compute_observations(ids)
        self._record_cached_sampling(ids)

    def _record_cached_sampling(self, ids):
        templates, pre = self._templates[ids], self._before_predecessor[ids]
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:7] += torch.bincount(templates.flatten(), minlength=6)
        self._stage2_domain_counts += torch.bincount(self._scene_domains[ids], minlength=3)
        self._size_rsi_executed[:, 0] += torch.bincount(templates.flatten(), minlength=6)
        self._count_sampling(self._scene_ids[ids], pre, templates, self._roles[ids], self._unreachable[ids])
        mode = self._bridge_start_mode[ids]
        self._bridge_start_samples += torch.bincount(mode*3+self._bridge_count[ids]-1, minlength=9).view(3, 3)
        before = self.relation_runtime.before[ids]
        for branch in range(3):
            chosen = self._bridge_branch[ids] == branch
            coupled = (templates[chosen] == 5) & (self._members[ids[chosen]] > 1)
            self._branch_samples[branch] += torch.stack((chosen.sum(), (pre[chosen] >= 0).sum(),
                before[chosen].sum(), self._crosser_mask[ids[chosen]].sum(), coupled.sum())).float()
        cross = self._roles[ids, 1]
        left = self._bridge_side[ids] == (self._start_surface[ids, cross] == 1)
        family = templates[torch.arange(len(ids), device=self.device), self._roles[ids, 3]]
        self._fixed_before_stats += torch.stack((left.sum(), (~left).sum(),
            *[(family == task).sum() for task in (1, 2, 4)])).float()

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        if self._layout_cache is not None:
            result['layout_cache/hits'] = self._layout_hits[0].clone()
            result['layout_cache/misses'] = self._layout_hits[1].clone()
            result['layout_cache/hit_fraction'] = self._layout_hits[0]/self._layout_hits.sum().clamp_min(1)
            self._layout_hits.zero_()
        return result
