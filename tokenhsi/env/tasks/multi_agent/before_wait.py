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
        self.climb_margin = config.get('climb_radius_buffer')
        self.coefficient = config.get('coefficient', 0.)
        self.ontop_distance_source = config.get('ontop_distance_source', 'payload')
        self.speed_threshold = config.get('speed_threshold', 0.)
        self.speed_dims = 3 if config.get('speed_axes', 'xy') == 'xyz' else 2
        self.cumulative = config.get('violation_mode', 'continuous') == 'cumulative'
        self.grace_steps = math.ceil(config.get('grace_seconds', 0.)/dt - 1e-6)
        self.violation_steps = math.ceil(config.get('moving_seconds', 1.)/dt - 1e-6)
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
        self.radius_buffer = self.margin
        if self.climb_margin is not None:
            self.radius_buffer = torch.where(relation[rows, edges] == CLIMB,
                self.climb_margin, self.margin)
        self.support = dst[rows, edges] - self.num_agents
        self.goal = dst[rows, self.predecessor] - self.num_agents - self.num_objects
        self.released[rows, edges] |= raw_success[rows, self.predecessor]

    def _observe(self, roots, objects, sizes, goals, raw_success):
        rows, edges = self.rows, self.edges
        m, o = self.num_agents, self.num_objects
        xy = roots[rows, self.owners, :2]
        if self.ontop_distance_source == 'payload':
            xy = torch.where(self.top[:, None],
                objects[rows, (self.source-m).clamp(0, o-1), :2], xy)
        distance = (xy-goals[rows, self.goal, :2]).norm(dim=-1)
        radius = sizes[rows, self.support, :2].norm(dim=-1)/2 + self.radius_buffer
        released = self.released[rows, edges] | raw_success[rows, self.predecessor]
        armed = (self.armed[rows, edges] | (distance <= radius)) & ~released
        self.released[rows, edges] = released
        self.armed[rows, edges] = armed
        return armed

    def penalty(self, roots, root_velocity, objects, sizes, goals, raw_success):
        armed = self._observe(roots, objects, sizes, goals, raw_success)
        penalty = torch.zeros(roots.shape[:2], dtype=roots.dtype, device=roots.device)
        penalty[self.rows, self.owners] = (self.coefficient * armed
            * root_velocity[self.rows, self.owners, :3].norm(dim=-1))
        return penalty

    def step(self, roots, root_velocity, objects, sizes, goals, raw_success):
        rows, edges = self.rows, self.edges
        armed = self._observe(roots, objects, sizes, goals, raw_success)
        monitor = armed & (self.age[rows, edges] >= self.grace_steps)
        moving = root_velocity[rows, self.owners, :self.speed_dims].norm(dim=-1) > self.speed_threshold
        if self.cumulative:
            count = torch.where(armed, self.moving_steps[rows, edges]+(monitor & moving).long(), 0)
        else:
            count = torch.where(monitor & moving, self.moving_steps[rows, edges]+1, 0)
        self.age[rows, edges] = torch.where(armed, self.age[rows, edges]+1, 0)
        self.moving_steps[rows, edges] = count
        terminate = torch.zeros(roots.shape[:2], dtype=torch.bool, device=roots.device)
        terminate[rows, self.owners] = count >= self.violation_steps
        return terminate
