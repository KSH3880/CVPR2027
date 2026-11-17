from collections import OrderedDict
from dataclasses import fields
import random
import time

import torch

from utils.edge_ontop_spec import copy_graph_rows, permute_graph, select_graph
from utils.sensor_stage2_spec import WAREHOUSE


STATE_FIELDS = ('_humanoid_root_states', '_kinematic_humanoid_rigid_body_states',
                '_dof_pos', '_dof_vel', '_every_env_init_dof_pos', '_box_states',
                '_platform_states', '_tar_platform_states', '_tar_pos', '_center_tar_pos')
CONTEXT_FIELDS = ('_groups', '_templates', '_before_predecessor', '_before_near_pairs',
                  '_members', '_regions', '_logical_box_order', '_agent_box_assignment',
                  '_surface', '_start_surface', '_roles', '_stage2_task_slots', '_scene_domains', '_unreachable')
KEY_FIELDS = ('_bridge_count', '_warehouse_direction', '_box_size', '_env_origins')


class BeforeLayoutCache:
    def __init__(self, num_envs, seed, capacity=32, samples=2, refresh_probability=.25, shuffle_edges=True, warmup_resets=0):
        if warmup_resets < 0:
            raise ValueError('BEFORE cache warmup resets must be nonnegative')
        self.entries = [OrderedDict() for _ in range(num_envs)]
        self.random = random.Random(seed)
        self.capacity, self.samples = capacity, samples
        self.refresh_probability, self.shuffle_edges = refresh_probability, shuffle_edges
        self.requests = self.hits = self.refreshes = 0
        self.keys, self.restored = {}, {}
        self.warmup_resets = warmup_resets
        self.warmed = warmup_resets == 0
        self.warmup_seconds = 0.

    def warmup(self, task, ids):
        if self.warmed:
            return
        selected = ids[task._scene_ids[ids] == WAREHOUSE]
        if not len(selected):
            return
        self.warmed = True
        flags = task._warehouse_before_pairs.clone()
        counters = {name: getattr(task, name).clone() for name in (
            '_sampling_counts', '_sampling_retries', '_scene_sampling', '_stage2_domain_counts',
            '_size_rsi_executed', '_relation_episode_id')}
        cpu_rng, cache_rng = torch.get_rng_state(), self.random.getstate()
        cuda = torch.device(task.device).type == 'cuda'
        cuda_rng = torch.cuda.get_rng_state(task.device) if cuda else None
        refresh = self.refresh_probability
        start = time.perf_counter()
        try:
            self.refresh_probability = 1.
            task._warehouse_before_pairs[selected] = True
            for _ in range(self.warmup_resets):
                task._reset_ontop_context_envs(selected)
            if cuda:
                torch.cuda.synchronize(task.device)
            self.warmup_seconds = time.perf_counter()-start
        finally:
            task._warehouse_before_pairs.copy_(flags)
            for name, value in counters.items():
                getattr(task, name).copy_(value)
            torch.set_rng_state(cpu_rng)
            if cuda:
                torch.cuda.set_rng_state(cuda_rng, task.device)
            self.random.setstate(cache_rng)
            self.refresh_probability = refresh
            self.requests = self.hits = self.refreshes = 0

    def restore(self, task, ids):
        self.keys, self.restored = {}, {}
        selected = ids[task._warehouse_before_pairs[ids]]
        if not len(selected):
            return selected
        condition = task._templates[selected]*2+task._before_near_pairs[selected].long()
        condition = condition.masked_fill(task._before_predecessor[selected] < 0, -1).sort(-1).values
        values = [condition.float()]
        values += [getattr(task, name)[selected].reshape(len(selected), -1).float() for name in KEY_FIELDS]
        values += [getattr(task, name)[selected, WAREHOUSE].reshape(len(selected), -1)
                   for name in ('_structure_yaw', '_structure_shift', '_surface_half')]
        keys = torch.cat(values, -1).cpu().tolist()
        self.requests += len(selected)
        for env, values in zip(selected.tolist(), keys):
            key = tuple(values)
            self.keys[env] = key
            entries = self.entries[env]
            if key not in entries:
                continue
            entries.move_to_end(key)
            if self.random.random() < self.refresh_probability:
                self.refreshes += 1
                continue
            self.restored[env] = self.random.choice(entries[key])
        hits = ids.new_tensor(list(self.restored))
        self.hits += len(hits)
        if len(hits):
            states = list(self.restored.values())
            for name in (*STATE_FIELDS, *CONTEXT_FIELDS):
                getattr(task, name)[hits] = torch.stack([state[name] for state in states])
            task.relation_runtime.before[hits] = torch.stack([state['before'] for state in states])
            graph = task.relation_runtime.graph
            for name in states[0]['graph']:
                getattr(graph, name)[hits] = torch.stack([state['graph'][name] for state in states])
            if self.shuffle_edges:
                chosen = select_graph(graph, hits)
                permutation = torch.rand_like(chosen.edge_src, dtype=torch.float).argsort(-1)
                copy_graph_rows(graph, hits, permute_graph(chosen, permutation))
            task._place_failed[hits] = False
        return hits

    def finish(self, task, fresh):
        if self.restored:
            hits = fresh.new_tensor(list(self.restored))
            task._hist_amp_obs_buf[hits] = torch.stack([state['_hist_amp_obs_buf'] for state in self.restored.values()])
        graph = task.relation_runtime.graph
        for env in fresh.tolist():
            if env not in self.keys:
                continue
            key, entries = self.keys[env], self.entries[env]
            state = {name: getattr(task, name)[env].detach().clone()
                     for name in (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')}
            state['before'] = task.relation_runtime.before[env].clone()
            state['graph'] = {field.name: getattr(graph, field.name)[env].clone() for field in fields(graph)
                              if isinstance(getattr(graph, field.name), torch.Tensor)}
            pool = entries.setdefault(key, [])
            pool.append(state)
            if len(pool) > self.samples:
                pool.pop(0)
            entries.move_to_end(key)
            if len(entries) > self.capacity:
                entries.popitem(last=False)
        self.keys, self.restored = {}, {}

    def diagnostics(self, task, result):
        values = dict(requests=self.requests, hits=self.hits, refreshes=self.refreshes,
                      hit_fraction=self.hits/max(self.requests, 1),
                      warmup_seconds=self.warmup_seconds,
                      entries=sum(len(pool) for entries in self.entries for pool in entries.values()))
        result.update({'warehouse/before_cache/'+name: torch.tensor(value, device=task.device)
                       for name, value in values.items()})
        self.requests = self.hits = self.refreshes = 0
