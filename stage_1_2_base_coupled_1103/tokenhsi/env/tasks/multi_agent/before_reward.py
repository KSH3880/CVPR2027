import torch

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success
from utils.edge_ontop_spec import select_graph, batched, ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_scenario_spec import paired_placement_success


def predecessor_at_progress(progress, roots, objects, sizes, goals, graph, config):
    """Retarget only dependent SIT/CLIMB/ON_TOP proximity, in either gate state.

    Keep the actual support's size/buffer and every non-dependent edge intact.
    Geometry for state/success remains object-relative in the ordinary evaluator.
    """
    n, m, o = len(roots), graph.num_agents, graph.num_objects
    relation, valid, src, dst = [batched(getattr(graph, name), n) for name in
        ('edge_relation', 'edge_valid', 'edge_src', 'edge_dst')]
    prereq = graph.prereq_mask
    if prereq.dim() == 2:
        prereq = prereq.unsqueeze(0).expand(n, -1, -1)
    dependent = valid & prereq.any(-1) & ((relation == SIT) | (relation == CLIMB) | (relation == ON_TOP))
    rows, edges = dependent.nonzero(as_tuple=True)
    predecessor_edge = prereq[rows, edges].long().argmax(-1)
    goal = dst[rows, predecessor_edge] - m - o
    source = src[rows, edges]
    top = relation[rows, edges] == ON_TOP
    source_xy = torch.where(top[:, None],
        objects[rows, (source-m).clamp(0, o-1), :2],
        roots[rows, source.clamp(0, m-1), :2])
    distance = (source_xy - goals[rows, goal, :2]).norm(dim=-1)
    p = config['progress']
    radius = torch.full_like(distance, float(p.get('delta', 0.)))
    if 'bbox_buffers' in p:
        buffers = p['bbox_buffers']
        radius = sizes[rows, dst[rows, edges]-m, :2].norm(dim=-1)/2
        radius += torch.where(top, float(buffers['ON_TOP']),
            torch.where(relation[rows, edges] == SIT, float(buffers['SIT']), float(buffers['CLIMB'])))
    result = progress.clone()
    result[rows, edges] = 1 / (1 + (distance-radius).clamp_min(0)/p['sigma'])
    return result


class BeforeRuntime(Stage1ContextRuntime):
    def _raw_success(self, phi, z_error, feet_error, region_error, graph):
        return interaction_own_success(phi, z_error, feet_error, graph, self.config, region_error)

    def components(self, phi, progress, z_error, feet_error, region_error, graph):
        raw = self._raw_success(phi, z_error, feet_error, region_error, graph)
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
