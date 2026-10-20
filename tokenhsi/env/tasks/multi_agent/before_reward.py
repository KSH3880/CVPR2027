import torch

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success
from utils.edge_ontop_spec import select_graph
from utils.edge_scenario_spec import paired_placement_success


class BeforeRuntime(Stage1ContextRuntime):
    def components(self, phi, progress, z_error, feet_error, region_error, graph):
        raw = interaction_own_success(phi, z_error, feet_error, graph, self.config, region_error)
        gate = (raw[:, None] | ~graph.prereq_mask).all(-1)
        success = raw & gate
        paired = paired_placement_success(success, graph)
        saturated = (success | paired) & graph.edge_valid
        state = torch.where(saturated, 1., phi)*gate*graph.edge_valid*self.config['state_reward_weight']
        prog = torch.where(saturated, 1., progress)*graph.edge_valid*self.config['progress_reward_weight']
        bonus = saturated.float()*self.config['success_reward_weight']
        total = state+prog+bonus
        local = owner_sum(total, graph)
        return dict(phi_raw=phi, progress_raw=progress, raw_own_success=raw,
            prerequisite_gate=gate, own_success=success, term_success=torch.zeros_like(success),
            reward_saturated=saturated, paired_placement_success=paired,
            state_component=state, progress_component=prog, success_component=bonus,
            total=total, local_task_reward=local, agent_task_reward=local,
            teammate_task_reward=torch.zeros_like(local))

    def reset(self, ids, phi, z_error, feet_error=None, region_error=None):
        graph = select_graph(self.graph, ids)
        result = self.components(phi, torch.zeros_like(phi), z_error, feet_error, region_error, graph)
        self.phi[ids] = phi
        self.own_success[ids] = result['own_success']
        self.achieved[ids] = result['own_success']
        self.done[ids] = goal_success(result['own_success'], graph)

    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        result = self.components(phi, progress, z_error, feet_error, region_error, self.graph)
        self.phi.copy_(phi)
        self.own_success.copy_(result['own_success'])
        self.achieved |= result['own_success']
        self.done.copy_(goal_success(result['own_success'], self.graph))
        self.last_result = result
        return result
