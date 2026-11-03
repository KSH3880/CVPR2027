import math

import torch

from utils.edge_ontop_spec import batched, ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB


class BeforeWait:
    def __init__(self, graph, num_envs, dt, config, device):
        shape = (num_envs, len(graph.ids))
        self.armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.released = torch.zeros_like(self.armed)
        self.age = torch.zeros(shape, dtype=torch.long, device=device)
        self.moving_steps = torch.zeros_like(self.age)
        self.margin = config['radius_buffer']
        self.speed_threshold = config['speed_threshold']
        self.grace_steps = math.ceil(config['grace_seconds']/dt - 1e-6)
        self.violation_steps = math.ceil(config['moving_seconds']/dt - 1e-6)
        self.num_agents = graph.num_agents
        self.num_objects = graph.num_objects

    def reset(self, ids, graph, raw_success):
        for value in (self.armed, self.released, self.age, self.moving_steps):
            value[ids] = 0
        n = len(self.armed)
        relation, valid, src, dst, owner = [batched(getattr(graph, name), n) for name in
            ('edge_relation', 'edge_valid', 'edge_src', 'edge_dst', 'edge_owner')]
        prereq = graph.prereq_mask
        if prereq.dim() == 2:
            prereq = prereq.unsqueeze(0).expand(n, -1, -1)
        dependent = valid & prereq.any(-1) & ((relation == SIT) | (relation == CLIMB) | (relation == ON_TOP))
        self.rows, self.edges = dependent.nonzero(as_tuple=True)
        rows, edges = self.rows, self.edges
        self.predecessor = prereq[rows, edges].long().argmax(-1)
        self.owners = owner[rows, edges]
        self.source = src[rows, edges]
        self.top = relation[rows, edges] == ON_TOP
        self.support = dst[rows, edges] - self.num_agents
        self.goal = dst[rows, self.predecessor] - self.num_agents - self.num_objects
        self.released[rows, edges] |= raw_success[rows, self.predecessor]

    def step(self, roots, root_velocity, objects, sizes, goals, raw_success):
        rows, edges = self.rows, self.edges
        m, o = self.num_agents, self.num_objects
        xy = torch.where(self.top[:, None],
            objects[rows, (self.source-m).clamp(0, o-1), :2],
            roots[rows, self.source.clamp(0, m-1), :2])
        distance = (xy-goals[rows, self.goal, :2]).norm(dim=-1)
        radius = sizes[rows, self.support, :2].norm(dim=-1)/2 + self.margin
        released = self.released[rows, edges] | raw_success[rows, self.predecessor]
        armed = (self.armed[rows, edges] | (distance <= radius)) & ~released
        monitor = armed & (self.age[rows, edges] >= self.grace_steps)
        moving = root_velocity[rows, self.owners, :2].norm(dim=-1) > self.speed_threshold
        count = torch.where(monitor & moving, self.moving_steps[rows, edges]+1, 0)
        self.released[rows, edges] = released
        self.armed[rows, edges] = armed
        self.age[rows, edges] = torch.where(armed, self.age[rows, edges]+1, 0)
        self.moving_steps[rows, edges] = count
        terminate = torch.zeros(roots.shape[:2], dtype=torch.bool, device=roots.device)
        terminate[rows, self.owners] = count >= self.violation_steps
        return terminate
