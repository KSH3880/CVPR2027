"""Stage1 local edge rewards, placement saturation and task observations."""

import torch
from push_reference.utils.task_role_spec import task_graph_packet
from push_reference.utils.edge_ontop_spec import select_graph
from push_reference.utils.edge_scenario_spec import paired_placement_success
from push_reference.env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success
from push_reference.env.tasks.multi_agent.edge_interaction_reward import interaction_own_success


class Stage1ContextRuntime:

    def __init__(self, num_envs, graph, config, device):
        self.graph, self.config = (graph, config)
        self.phi = torch.zeros(num_envs, len(graph.ids), device=device)
        self.own_success = torch.zeros_like(self.phi, dtype=torch.bool)
        self.achieved = torch.zeros_like(self.own_success)
        self.done = torch.zeros(
            num_envs, graph.num_agents, dtype=torch.bool, device=device
        )
        self.last_result = None

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        graph = select_graph(self.graph, ids)
        self.phi[ids] = phi
        self.own_success[ids] = interaction_own_success(
            phi, z_error, feet_error, graph, self.config, region_error
        )
        self.achieved[ids] = self.own_success[ids]
        self.done[ids] = goal_success(self.own_success[ids], graph)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        if feet_error is None:
            feet_error = torch.zeros_like(phi)
        success = interaction_own_success(
            phi, z_error, feet_error, self.graph, self.config, region_error
        )
        paired = paired_placement_success(success, self.graph)
        saturated = (success | paired) & self.graph.edge_valid
        state = (
            torch.where(saturated, 1.0, phi)
            * self.config["state_reward_weight"]
            * self.graph.edge_valid
        )
        prog = (
            torch.where(saturated, 1.0, progress)
            * self.config["progress_reward_weight"]
            * self.graph.edge_valid
        )
        bonus = saturated.to(phi.dtype) * self.config["success_reward_weight"]
        total = state + prog + bonus
        local = owner_sum(total, self.graph)
        teammate = torch.zeros_like(local)
        result = dict(
            phi_raw=phi,
            progress_raw=progress,
            own_success=success,
            term_success=torch.zeros_like(success),
            reward_saturated=saturated,
            paired_placement_success=paired,
            state_component=state,
            progress_component=prog,
            success_component=bonus,
            total=total,
            teammate_task_reward=teammate,
            agent_task_reward=local,
            local_task_reward=local,
        )
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result

    def suffix(self, ids=None):
        graph = self.graph if ids is None else select_graph(self.graph, ids)
        return task_graph_packet(
            graph, self.phi.shape[0] if ids is None else self.phi[ids].shape[0]
        )
