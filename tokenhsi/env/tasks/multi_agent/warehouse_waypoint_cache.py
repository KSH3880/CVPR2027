"""Batched device-resident layout cache, separated by Warehouse variant.

Only initial layouts are cached, never episode success or waypoint latches.
Role sampling precedes lookup, so a miss cannot change the episode mixture.
"""
from dataclasses import fields
import time
import torch
from env.tasks.multi_agent.warehouse_before_cache import STATE_FIELDS, CONTEXT_FIELDS, KEY_FIELDS
from utils.edge_ontop_spec import copy_graph_rows, select_graph, permute_graph
from utils.sensor_stage2_spec import WAREHOUSE


class WaypointLayoutCache:
    def __init__(self, task, cfg):
        self.capacity, self.samples = cfg['capacity'], cfg['samples']
        self.refresh_probability = cfg['refresh_probability']
        self.warmup_resets, self.warmed = cfg['warmup_resets'], False
        self.warmup_seconds = 0.
        self.data = {}
        shape = (task.num_envs, 2, self.capacity)
        self.count = torch.zeros(shape, device=task.device, dtype=torch.long)
        self.age = torch.zeros_like(self.count)
        self.pending_slot = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
        self.clock = 0
        self.stats = torch.zeros(2, 3, device=task.device)
        self.hits = torch.empty(0, dtype=torch.long, device=task.device)
        self.keys = None

    def warmup(self, task, ids):
        if self.warmed:
            return
        self.warmed = True
        names = ('_warehouse_before_pairs', '_waypoint_phase', '_sampling_counts', '_sampling_retries',
                 '_scene_sampling', '_stage2_domain_counts', '_size_rsi_executed', '_relation_episode_id')
        saved = {name: getattr(task, name).clone() for name in names}
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(task.device) if str(task.device).startswith('cuda') else None
        refresh = self.refresh_probability
        compute_observations = getattr(task, '_compute_observations', None)
        start = time.perf_counter()
        try:
            self.refresh_probability = 1.
            # Warmup stores physical layouts and AMP history, never policy observations.
            # Avoid 64 discarded 48x48 sensor renders and transformer metadata updates.
            if compute_observations is not None:
                task._compute_observations = lambda env_ids=None: None
            for variant in (True, False):
                task._warehouse_before_pairs[ids] = variant
                for _ in range(self.warmup_resets):
                    task._waypoint_phase[ids] = 0
                    task._reset_ontop_context_envs(ids)
            if cuda is not None:
                torch.cuda.synchronize(task.device)
            self.warmup_seconds = time.perf_counter()-start
        finally:
            if compute_observations is not None:
                task._compute_observations = compute_observations
            for name, value in saved.items():
                getattr(task, name).copy_(value)
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state(cuda, task.device)
            self.refresh_probability = refresh
            self.stats.zero_()

    def _key(self, task, ids):
        # The apron cache retains its original unordered follower-task/near key.
        pre = task._before_predecessor[ids]
        condition = (task._templates[ids]*2+task._before_near_pairs[ids].long()).masked_fill(pre < 0, -1).sort(-1).values
        exact = torch.cat((task._templates[ids], pre, task._before_near_pairs[ids].long(),
                           task._regions[ids]), -1).float()
        canonical = torch.zeros_like(exact)
        canonical[:, :condition.shape[1]] = condition
        values = [torch.where(task._warehouse_before_pairs[ids, None], canonical, exact)]
        values += [getattr(task, name)[ids].reshape(len(ids), -1).float() for name in KEY_FIELDS]
        values += [getattr(task, name)[ids, WAREHOUSE].reshape(len(ids), -1)
                   for name in ('_structure_yaw', '_structure_shift', '_surface_half')]
        values.append(task._warehouse_start_mode[ids, None].float())
        return torch.cat(values, -1)

    def restore(self, task, ids):
        self.clock += 1
        variant = task._warehouse_before_pairs[ids].long()
        keys = self._key(task, ids)
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
        # Reserve/replace keys only for fresh rows. No CPU key construction or per-env loop.
        missed = ~found
        self.keys[ids[missed], variant[missed], slot[missed]] = keys[missed]
        self.count[ids[missed], variant[missed], slot[missed]] = 0
        if len(selected):
            counts = self.count[selected, kind, entry].clamp_max(self.samples)
            sample = (torch.rand(len(selected), device=task.device)*counts).long()
            self.hit_indices = (selected, kind, entry, sample)
            for name, bank in self.data.items():
                value = bank[selected, kind, entry, sample]
                if name.startswith('graph:'):
                    getattr(task.relation_runtime.graph, name[6:])[selected] = value
                elif name == 'before':
                    task.relation_runtime.before[selected] = value
                else:
                    getattr(task, name)[selected] = value
            if task._relation_graph_spec['shuffle_edge_order']:
                graph = task.relation_runtime.graph
                chosen = select_graph(graph, selected)
                order = torch.rand_like(chosen.edge_src, dtype=torch.float).argsort(-1)
                copy_graph_rows(graph, selected, permute_graph(chosen, order))
            task._place_failed[selected] = False
        return selected

    def finish(self, task, fresh):
        # _init_amp_obs runs on all reset rows; put the validated histories back for hits.
        if len(self.hits):
            task._hist_amp_obs_buf[self.hits] = self.data['_hist_amp_obs_buf'][self.hit_indices]
        if not len(fresh):
            return
        variant = task._warehouse_before_pairs[fresh].long()
        slot = self.pending_slot[fresh]
        sample = self.count[fresh, variant, slot] % self.samples
        values = {name: getattr(task, name) for name in (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')}
        if hasattr(task, '_warehouse_landing'):
            values['_warehouse_landing'] = task._warehouse_landing
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
        for index, variant in enumerate(('push_before', 'before_pairs')):
            prefix = 'warehouse/layout_cache/'+variant+'/'
            for column, name in enumerate(('requests', 'hits', 'refreshes')):
                result[prefix+name] = self.stats[index, column].clone()
            result[prefix+'hit_fraction'] = self.stats[index, 1]/self.stats[index, 0].clamp_min(1)
        result['warehouse/layout_cache/warmup_seconds'] = torch.tensor(self.warmup_seconds, device=task.device)
        result['warehouse/layout_cache/bytes'] = torch.tensor(sum(v.numel()*v.element_size() for v in self.data.values()), device=task.device)
        self.stats.zero_()
