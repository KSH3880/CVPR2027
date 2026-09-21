"""Stage-1 own-edge rewards and constant START/KEEP observations."""
import torch

from utils.edge_ontop_spec import graph_packet, select_graph
from env.tasks.multi_agent.edge_context_reward import edge_context_reward, goal_success
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_ontop_reward import OnTopContextRuntime


def stage1_context(phi, graph):
    valid = graph.edge_valid
    if valid.ndim == 1:
        valid = valid.unsqueeze(0).expand(phi.shape[0], -1)
    return torch.ones(*phi.shape, 2, device=phi.device, dtype=phi.dtype) * valid[..., None]


class Stage1ContextRuntime(OnTopContextRuntime):
    def reset(self, ids, phi, z_error, feet_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        graph = select_graph(self.graph, ids)
        self.phi[ids] = phi
        self.own_success[ids] = interaction_own_success(
            phi, z_error, feet_error, graph, self.config)
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], graph)

    def step(self, phi, progress, z_error, feet_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        success = interaction_own_success(
            phi, z_error, feet_error, self.graph, self.config)
        # All term indices are -1, so the shared kernel saturates on own success only.
        result = edge_context_reward(phi, progress, success, self.graph, self.config)
        result['local_task_reward'] = result['agent_task_reward']
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result

    def suffix(self, ids=None):
        graph = self.graph if ids is None else select_graph(self.graph, ids)
        phi = self.phi if ids is None else self.phi[ids]
        return graph_packet(graph, stage1_context(phi, graph))
