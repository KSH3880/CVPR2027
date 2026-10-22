"""Original Stage1 AT and existing tasks, with a separately weighted CONTACT edge."""
import torch
from utils.edge_ontop_spec import batched
from utils.edge_context_spec import AT
from utils.edge_scenario_spec import paired_placement_success
from utils.stage1_push_spec import PUSH
from env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime


class Stage1PushRuntime(Stage1ContextRuntime):
    def __init__(self, *args, contact_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_weights = contact_weights
        self.contact = torch.zeros_like(self.done)

    def _success(self, phi, z_error, feet_error, region_error, graph=None, contact=None):
        graph = self.graph if graph is None else graph
        success = interaction_own_success(phi, z_error, feet_error, graph, self.config, region_error)
        owner = batched(graph.edge_owner, phi.shape[0])
        touch = self.contact if contact is None else contact
        return torch.where(batched(graph.edge_relation, phi.shape[0]) == PUSH,
                           touch.gather(1, owner), success) & graph.edge_valid

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        from utils.edge_ontop_spec import select_graph
        self.contact[ids] = False
        success = self._success(phi, z_error, feet_error, region_error,
                                select_graph(self.graph, ids), self.contact[ids])
        self.phi[ids] = phi
        self.own_success[ids] = success
        self.achieved[ids] = success
        self.done[ids] = goal_success(success, select_graph(self.graph, ids))

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        success = self._success(phi, z_error, feet_error, region_error)
        paired = paired_placement_success(success, self.graph)
        saturated = (success | paired) & self.graph.edge_valid
        push = batched(self.graph.edge_relation, phi.shape[0]) == PUSH
        components = []
        for index, (raw, key) in enumerate(((phi, 'state_reward_weight'),
                                          (progress, 'progress_reward_weight'),
                                          (torch.zeros_like(phi), 'success_reward_weight'))):
            weight = torch.where(push, self.contact_weights[index], self.config[key])
            value = torch.where(saturated, 1., raw) * weight * self.graph.edge_valid
            components.append(value)
        state, prog, bonus = components
        local = owner_sum(state + prog + bonus, self.graph)
        result = dict(phi_raw=phi, progress_raw=progress, own_success=success,
            term_success=torch.zeros_like(success), reward_saturated=saturated,
            paired_placement_success=paired, state_component=state,
            progress_component=prog, success_component=bonus, total=state+prog+bonus,
            teammate_task_reward=torch.zeros_like(local), agent_task_reward=local,
            local_task_reward=local)
        self.phi.copy_(phi)
        self.own_success.copy_(success)
        self.achieved |= success
        self.done.copy_(goal_success(success, self.graph))
        self.last_result = result
        return result
