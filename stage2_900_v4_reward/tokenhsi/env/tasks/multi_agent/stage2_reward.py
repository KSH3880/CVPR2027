import math

import torch

from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from utils.edge_ontop_spec import batched


class Stage2Runtime(BeforePushRuntime):
    def __init__(self, *args, before_at_z_tolerance=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_at_z_tolerance = before_at_z_tolerance
        self.gate = torch.ones_like(self.done)
        self.before = torch.zeros_like(self.done)
        self._reset_ids = None

    def _raw_success(self, phi, z_error, feet_error, region_error, graph):
        success = super()._raw_success(phi, z_error, feet_error, region_error, graph)
        before = self.before if self._reset_ids is None else self.before[self._reset_ids]
        before_at = before.gather(1, batched(graph.edge_owner, len(phi))) & (graph.edge_relation == 7)
        threshold = math.exp(-self.config['at']['near_distance_scale'] * self.config['at']['success_distance']**2)
        relaxed = (phi >= threshold) & (z_error.abs() <= self.before_at_z_tolerance) & graph.edge_valid
        return torch.where(before_at, relaxed, success)

    def _gated(self, value, graph, gate):
        active = (graph.edge_relation != 7) | gate.gather(1, graph.edge_owner)
        return value * active

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        from utils.edge_ontop_spec import select_graph
        graph = select_graph(self.graph, ids)
        self._reset_ids = ids
        try:
            return super().reset(ids, self._gated(phi, graph, self.gate[ids]),
                                 z_error, feet_error, region_error)
        finally:
            self._reset_ids = None

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        result = super().step(self._gated(phi, self.graph, self.gate),
            self._gated(progress, self.graph, self.gate), z_error, feet_error, region_error)
        result['group_gate'] = self.gate
        return result
