"""Training-only GPU cache of validated Warehouse initial layouts (ported from warehouse_fixed's waypoint cache).

Each env keeps two banks: the regular 2 PUSH + BEFORE layout and the config's special start (BEFORE pairs or
leader-done). Roles and variants are sampled before lookup, so a miss cannot change the episode mixture.
Only initial layouts and AMP histories are cached, never episode outcomes.
"""
from dataclasses import fields
import time

import torch

from utils.edge_ontop_spec import copy_graph_rows, permute_graph, select_graph
from utils.sensor_stage2_spec import WAREHOUSE

STATE_FIELDS = ('_humanoid_root_states', '_kinematic_humanoid_rigid_body_states',
                '_dof_pos', '_dof_vel', '_every_env_init_dof_pos', '_box_states',
                '_platform_states', '_tar_platform_states', '_tar_pos', '_center_tar_pos', '_warehouse_landing')
CONTEXT_FIELDS = ('_groups', '_templates', '_before_predecessor', '_before_near_pairs',
                  '_members', '_regions', '_logical_box_order', '_agent_box_assignment',
                  '_surface', '_start_surface', '_roles', '_stage2_task_slots', '_scene_domains', '_unreachable')
KEY_FIELDS = ('_bridge_count', '_warehouse_direction', '_box_size', '_env_origins')
COUNTERS = ('_sampling_counts', '_sampling_retries', '_scene_sampling', '_stage2_domain_counts',
            '_size_rsi_executed', '_relation_episode_id', '_warehouse_start_mode', '_warehouse_start_samples',
            '_warehouse_coop_samples')


class WarehouseLayoutCache:
    def __init__(self, task, cfg):
        self.capacity, self.samples = cfg['capacity'], cfg['samples']
        self.refresh_probability = cfg['refresh_probability']
        self.warmup_resets, self.warmed = cfg['warmup_resets'], cfg['warmup_resets'] == 0
        self.warmup_seconds = 0.
        if task._warehouse_before_probability:
            self.flag, self.names = '_warehouse_before_pairs', ('push_before', 'before_pairs')
        elif task._warehouse_done_probability:
            self.flag, self.names = '_warehouse_leader_done', ('main', 'leader_done')
        else:
            self.flag, self.names = None, ('main',)
        self.data = {}
        self.count = torch.zeros(task.num_envs, len(self.names), self.capacity, device=task.device, dtype=torch.long)
        self.age = torch.zeros_like(self.count)
        self.pending_slot = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
        self.clock = 0
        self.stats = torch.zeros(len(self.names), 3, device=task.device)
        self.hits = torch.empty(0, dtype=torch.long, device=task.device)
        self.keys = None

    def _variant(self, task, ids):
        if self.flag is None:
            return torch.zeros_like(ids)
        return getattr(task, self.flag)[ids].long()

    def warmup(self, task, ids):
        if self.warmed:
            return
        ids = ids[task._scene_ids[ids] == WAREHOUSE]
        if not len(ids):
            return
        self.warmed = True
        from env.tasks.multi_agent.warehouse_sensor import sample_starts
        names = COUNTERS+((self.flag,) if self.flag else ())
        saved = {name: getattr(task, name).clone() for name in names if hasattr(task, name)}
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(task.device) if str(task.device).startswith('cuda') else None
        refresh = self.refresh_probability
        compute_observations = task._compute_observations
        start = time.perf_counter()
        try:
            self.refresh_probability = 1.
            # Warmup stores physical layouts and AMP histories only; skip the discarded sensor observations.
            task._compute_observations = lambda env_ids=None: None
            for variant in range(len(self.names)):
                if self.flag:
                    getattr(task, self.flag)[ids] = bool(variant)
                for _ in range(self.warmup_resets):
                    if self.flag == '_warehouse_leader_done' and variant:
                        task._warehouse_start_mode[ids] = 2
                    elif task._warehouse_start:
                        sample_starts(task, ids)
                    task._reset_ontop_context_envs(ids)
            if cuda is not None:
                torch.cuda.synchronize(task.device)
            self.warmup_seconds = time.perf_counter()-start
        finally:
            task._compute_observations = compute_observations
            for name, value in saved.items():
                getattr(task, name).copy_(value)
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state(cuda, task.device)
            self.refresh_probability = refresh
            self.stats.zero_()

    def _key(self, task, ids, variant):
        pre = task._before_predecessor[ids]
        exact = torch.cat((task._templates[ids], pre, task._before_near_pairs[ids].long(), task._regions[ids]), -1).float()
        start = task._warehouse_start_mode[ids].float()
        if self.flag == '_warehouse_before_pairs':
            # Apron teams keep the original unordered follower-task/near key; the door start is unused there.
            condition = (task._templates[ids]*2+task._before_near_pairs[ids].long()).masked_fill(pre < 0, -1).sort(-1).values
            canonical = torch.zeros_like(exact)
            canonical[:, :condition.shape[1]] = condition
            exact = torch.where(variant[:, None] > 0, canonical, exact)
            start = start.masked_fill(variant > 0, 0)
        values = [exact]
        values += [getattr(task, name)[ids].reshape(len(ids), -1).float() for name in KEY_FIELDS]
        values += [getattr(task, name)[ids, WAREHOUSE].reshape(len(ids), -1)
                   for name in ('_structure_yaw', '_structure_shift', '_surface_half')]
        values.append(start[:, None])
        return torch.cat(values, -1)

    def restore(self, task, ids):
        ids = ids[task._scene_ids[ids] == WAREHOUSE]
        self.hits = ids[:0]
        if not len(ids):
            return self.hits
        self.clock += 1
        variant = self._variant(task, ids)
        keys = self._key(task, ids, variant)
        if self.keys is None:
            self.keys = keys.new_zeros(*self.count.shape, keys.shape[-1])
        match = (self.keys[ids, variant] == keys[:, None]).all(-1) & (self.count[ids, variant] > 0)
        found = match.any(-1)
        slot = torch.where(found, match.long().argmax(-1), self.age[ids, variant].argmin(-1))
        self.pending_slot[ids] = slot
        refresh = torch.rand(len(ids), device=task.device) < self.refresh_probability
        hit = found & ~refresh
        selected, kind, entry = ids[hit], variant[hit], slot[hit]
        self.hits = selected
        for column, mask in enumerate((torch.ones_like(hit), hit, found & refresh)):
            self.stats[:, column].index_add_(0, variant, mask.float())
        self.age[ids, variant, slot] = self.clock
        missed = ~found
        self.keys[ids[missed], variant[missed], slot[missed]] = keys[missed]
        self.count[ids[missed], variant[missed], slot[missed]] = 0
        if len(selected):
            counts = self.count[selected, kind, entry].clamp_max(self.samples)
            sample = (torch.rand(len(selected), device=task.device)*counts).long()
            self.hit_indices = (selected, kind, entry, sample)
            graph = task.relation_runtime.graph
            for name, bank in self.data.items():
                value = bank[selected, kind, entry, sample]
                if name.startswith('graph:'):
                    getattr(graph, name[6:])[selected] = value
                elif name == 'before':
                    task.relation_runtime.before[selected] = value
                else:
                    getattr(task, name)[selected] = value
            if task._relation_graph_spec['shuffle_edge_order']:
                chosen = select_graph(graph, selected)
                order = torch.rand_like(chosen.edge_src, dtype=torch.float).argsort(-1)
                copy_graph_rows(graph, selected, permute_graph(chosen, order))
            task._place_failed[selected] = False
        return selected

    def finish(self, task, fresh):
        # _init_amp_obs runs on every reset row; put the validated histories back for hits.
        if len(self.hits):
            task._hist_amp_obs_buf[self.hits] = self.data['_hist_amp_obs_buf'][self.hit_indices]
        fresh = fresh[task._scene_ids[fresh] == WAREHOUSE]
        if not len(fresh):
            return
        variant = self._variant(task, fresh)
        slot = self.pending_slot[fresh]
        sample = self.count[fresh, variant, slot] % self.samples
        values = {name: getattr(task, name) for name in (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')}
        values['before'] = task.relation_runtime.before
        graph = task.relation_runtime.graph
        values.update({'graph:'+f.name: getattr(graph, f.name) for f in fields(graph)
                       if isinstance(getattr(graph, f.name), torch.Tensor)})
        for name, value in values.items():
            if name not in self.data:
                self.data[name] = value.new_empty(*self.count.shape, self.samples, *value.shape[1:])
            self.data[name][fresh, variant, slot, sample] = value[fresh]
        self.count[fresh, variant, slot] += 1

    def diagnostics(self, task, result):
        for index, variant in enumerate(self.names):
            prefix = 'warehouse/layout_cache/'+variant+'/'
            for column, name in enumerate(('requests', 'hits', 'refreshes')):
                result[prefix+name] = self.stats[index, column].clone()
            result[prefix+'hit_fraction'] = self.stats[index, 1]/self.stats[index, 0].clamp_min(1)
        result['warehouse/layout_cache/warmup_seconds'] = torch.tensor(self.warmup_seconds, device=task.device)
        result['warehouse/layout_cache/bytes'] = torch.tensor(
            float(sum(v.numel()*v.element_size() for v in self.data.values())), device=task.device)
        self.stats.zero_()
