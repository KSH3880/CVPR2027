"""Stage-1 own-edge rewards and constant START/KEEP observations."""
import torch
from utils.task_role_spec import TASK_ROLE_VARIANT, task_graph_packet

from utils.edge_ontop_spec import batched, graph_packet, select_graph
from env.tasks.multi_agent.edge_context_reward import edge_context_reward, goal_success, owner_sum
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_ontop_reward import OnTopContextRuntime
from utils.edge_stage1_spec import (OWNER_HOLDING_VARIANT, OWNER_HOLDING_VARIANTS,
    owner_holding_graph_packet, semantic_graph_packet)


def stage1_context(phi, graph):
    valid = graph.edge_valid
    if valid.ndim == 1:
        valid = valid.unsqueeze(0).expand(phi.shape[0], -1)
    return torch.ones(*phi.shape, 2, device=phi.device, dtype=phi.dtype) * valid[..., None]


class Stage1ContextRuntime(OnTopContextRuntime):
    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        graph = select_graph(self.graph, ids)
        self.phi[ids] = phi
        self.own_success[ids] = interaction_own_success(
            phi, z_error, feet_error, graph, self.config, region_error)
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], graph)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        success = interaction_own_success(
            phi, z_error, feet_error, self.graph, self.config, region_error)
        if self.config['schema_version'] in (8, 9, 10):
            from utils.edge_scenario_spec import paired_placement_success
            paired = paired_placement_success(success, self.graph)
            saturated = (success | paired) & self.graph.edge_valid
            state = torch.where(saturated, 1., phi) * self.config['state_reward_weight'] * self.graph.edge_valid
            prog = torch.where(saturated, 1., progress) * self.config['progress_reward_weight'] * self.graph.edge_valid
            bonus = saturated.to(phi.dtype) * self.config['success_reward_weight']
            total = state + prog + bonus
            local = owner_sum(total, self.graph)
            teammate_local = local
            aggregation = self.config.get('edge_aggregation')
            if aggregation in ('mean_active', 'self_sum_teammate_mean'):
                edge_count = owner_sum(batched(self.graph.edge_valid, phi.shape[0]).to(phi.dtype), self.graph)
                if aggregation == 'mean_active':
                    local = local / edge_count.clamp_min(1)
                    teammate_local = local
                else:
                    teammate_local = local / edge_count.clamp_min(1)
            sharing = self.config['task_sharing']
            teammate = (teammate_local.sum(-1, keepdim=True) - teammate_local) / (local.shape[-1] - 1) \
                if self.config['schema_version'] == 10 and local.shape[-1] > 2 else teammate_local.flip(-1)
            if sharing['teammate'] == 0:
                teammate = torch.zeros_like(local)
            result = dict(phi_raw=phi, progress_raw=progress, own_success=success,
                term_success=torch.zeros_like(success), reward_saturated=saturated,
                paired_placement_success=paired, state_component=state,
                progress_component=prog, success_component=bonus, total=total,
                teammate_task_reward=teammate,
                agent_task_reward=sharing['self'] * local +
                    sharing['teammate'] * teammate)
        else:
            result = edge_context_reward(phi, progress, success, self.graph, self.config)
        result['local_task_reward'] = (local if self.config['schema_version'] in (8, 9, 10)
                                       else result['agent_task_reward'])
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result

    def suffix(self, ids=None):
        graph = self.graph if ids is None else select_graph(self.graph, ids)
        phi = self.phi if ids is None else self.phi[ids]
        if self.config.get('stage1_variant') == TASK_ROLE_VARIANT:
            return task_graph_packet(graph, phi.shape[0])
        if self.config.get('stage1_variant') in OWNER_HOLDING_VARIANTS:
            return owner_holding_graph_packet(graph, phi)
        if self.config['schema_version'] in (7, 8, 9, 10):
            return semantic_graph_packet(graph, phi.shape[0]).to(phi.device)
        return graph_packet(graph, stage1_context(phi, graph))
