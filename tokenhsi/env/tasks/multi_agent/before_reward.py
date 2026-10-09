import torch

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_context_reward import goal_success, owner_sum
from utils.edge_scenario_spec import paired_placement_success
from utils.edge_ontop_spec import select_graph


def gated_success(raw, graph):
    gate = (raw[:, None, :] | ~graph.prereq_mask).all(-1)
    return raw & gate, gate


class BeforeRuntime(Stage1ContextRuntime):
    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        graph = select_graph(self.graph, ids)
        raw = interaction_own_success(phi, z_error, feet_error, graph, self.config, region_error)
        success, _ = gated_success(raw, graph)
        self.phi[ids] = phi
        self.own_success[ids] = success
        self.achieved[ids] = success
        self.done[ids] = goal_success(success, graph)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        raw = interaction_own_success(phi, z_error, feet_error, self.graph, self.config, region_error)
        success, gate = gated_success(raw, self.graph)
        paired = paired_placement_success(success, self.graph)
        saturated = (success | paired) & self.graph.edge_valid
        state = torch.where(saturated, 1., phi) * gate * self.graph.edge_valid * self.config['state_reward_weight']
        prog = torch.where(saturated, 1., progress) * self.graph.edge_valid * self.config['progress_reward_weight']
        bonus = saturated.to(phi.dtype) * self.config['success_reward_weight']
        total = state + prog + bonus
        local = owner_sum(total, self.graph)
        result = dict(phi_raw=phi, progress_raw=progress, own_success=success,
            raw_own_success=raw, prerequisite_gate=gate,
            term_success=torch.zeros_like(success), reward_saturated=saturated,
            paired_placement_success=paired, state_component=state,
            progress_component=prog, success_component=bonus, total=total,
            local_task_reward=local, agent_task_reward=local,
            teammate_task_reward=torch.zeros_like(local))
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved.copy_(success)
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result
