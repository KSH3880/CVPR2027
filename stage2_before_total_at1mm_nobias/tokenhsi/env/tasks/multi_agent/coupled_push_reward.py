import torch

from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime
from utils.edge_ontop_spec import select_graph


class CoupledPushRuntime(Stage1PushRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = torch.ones_like(self.done)

    def _gated(self, value, graph, gate):
        active = (graph.edge_relation != 7) | gate.gather(1, graph.edge_owner)
        return value*active

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        graph = select_graph(self.graph, ids)
        return super().reset(ids, self._gated(phi, graph, self.gate[ids]), z_error, feet_error, region_error)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        result = super().step(self._gated(phi, self.graph, self.gate),
            self._gated(progress, self.graph, self.gate), z_error, feet_error, region_error)
        result['group_gate'] = self.gate
        return result

