import torch

from utils.edge_ontop_spec import batched


class Stage2Wait:
    def __init__(self, graph, num_envs, option, device):
        self.armed = torch.zeros(num_envs, len(graph.ids), dtype=torch.bool, device=device)
        self.released = torch.zeros_like(self.armed)
        self.margin = option['radius_buffer']
        self.coefficient = option['coefficient']
        self.ontop_distance_source = option.get('ontop_distance_source', 'human')
        self.num_agents, self.num_objects = graph.num_agents, graph.num_objects

    def reset(self, ids, graph, raw_success):
        self.armed[ids] = False
        self.released[ids] = False
        n = len(self.armed)
        relation, valid, dst, owner = [batched(getattr(graph, name), n) for name in
                                     ('edge_relation', 'edge_valid', 'edge_dst', 'edge_owner')]
        dependent = valid & graph.prereq_mask.any(-1) & (
            (relation == 9) | (relation == 10) | (relation == 8))
        self.rows, self.edges = dependent.nonzero(as_tuple=True)
        rows, edges = self.rows, self.edges
        self.predecessor = graph.prereq_mask[rows, edges].long().argmax(-1)
        self.owners = owner[rows, edges]
        self.source = batched(graph.edge_src, n)[rows, edges]
        self.top = relation[rows, edges] == 8
        self.support = dst[rows, edges] - graph.num_agents
        self.goal = dst[rows, self.predecessor] - graph.num_agents - graph.num_objects
        self.released[rows, edges] |= raw_success[rows, self.predecessor]

    def penalty(self, roots, velocity, objects, sizes, goals, raw_success):
        rows, edges = self.rows, self.edges
        xy = roots[rows, self.owners, :2]
        if self.ontop_distance_source == 'payload':
            xy = torch.where(self.top[:, None],
                objects[rows, (self.source-self.num_agents).clamp(0, self.num_objects-1), :2], xy)
        distance = (xy-goals[rows, self.goal, :2]).norm(dim=-1)
        radius = sizes[rows, self.support, :2].norm(dim=-1)/2 + self.margin
        released = self.released[rows, edges] | raw_success[rows, self.predecessor]
        armed = (self.armed[rows, edges] | (distance <= radius)) & ~released
        self.released[rows, edges], self.armed[rows, edges] = released, armed
        penalty = torch.zeros(roots.shape[:2], dtype=roots.dtype, device=roots.device)
        penalty[rows, self.owners] = self.coefficient * armed * velocity[rows, self.owners, :3].norm(dim=-1)
        return penalty
