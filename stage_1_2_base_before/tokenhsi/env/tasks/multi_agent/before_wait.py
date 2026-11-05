import math

import torch

from utils.edge_ontop_spec import batched, ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_context_spec import HOLDING


class BeforeWait:
    def __init__(self, graph, num_envs, dt, config, device):
        shape = (num_envs, len(graph.ids))
        self.armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.released = torch.zeros_like(self.armed)
        self.picked_up = torch.zeros_like(self.armed)
        self.age = torch.zeros(shape, dtype=torch.long, device=device)
        self.moving_steps = torch.zeros_like(self.age)
        self.margin = config['radius_buffer']
        self.coefficient = config.get('coefficient', 0.)
        self.target_velocity_config = config.get('target_velocity')
        self.soft_entry_width = config.get('soft_entry_width', 0.)
        self.entry_weight = torch.zeros(shape, device=device)
        self.speed_threshold = config.get('speed_threshold', 0.)
        self.speed_dims = 3 if config.get('speed_axes', 'xy') == 'xyz' else 2
        self.cumulative = config.get('violation_mode', 'continuous') == 'cumulative'
        self.grace_steps = math.ceil(config.get('grace_seconds', 0.)/dt - 1e-6)
        self.violation_steps = math.ceil(config.get('moving_seconds', 1.)/dt - 1e-6)
        self.num_agents = graph.num_agents
        self.num_objects = graph.num_objects

    def reset(self, ids, graph, raw_success):
        for value in (self.armed, self.released, self.age, self.moving_steps, self.entry_weight, self.picked_up):
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
        if self.target_velocity_config is not None and 'ontop' in self.target_velocity_config:
            # Match by actor AND payload identity; shuffled graph order is irrelevant.
            match = (valid[rows] & (relation[rows] == HOLDING)
                     & (src[rows] == self.owners[:, None])
                     & (dst[rows] == self.source[:, None]))
            if not (match.sum(-1)[self.top] == 1).all():
                raise ValueError('ON_TOP velocity requires exactly one matching HOLDING edge')
            self.holding_edge = match.long().argmax(-1)
            reset_pair = (rows[:, None] == ids[None]).any(-1) & self.top
            r, e = rows[reset_pair], edges[reset_pair]
            self.picked_up[r, e] = raw_success[r, self.holding_edge[reset_pair]]

    def _observe(self, roots, objects, sizes, goals, raw_success):
        rows, edges = self.rows, self.edges
        m, o = self.num_agents, self.num_objects
        xy = torch.where(self.top[:, None],
            objects[rows, (self.source-m).clamp(0, o-1), :2],
            roots[rows, self.source.clamp(0, m-1), :2])
        distance = (xy-goals[rows, self.goal, :2]).norm(dim=-1)
        radius = sizes[rows, self.support, :2].norm(dim=-1)/2 + self.margin
        released = self.released[rows, edges] | raw_success[rows, self.predecessor]
        armed = (self.armed[rows, edges] | (distance <= radius)) & ~released
        if self.soft_entry_width:
            u = ((radius + self.soft_entry_width - distance) / self.soft_entry_width).clamp(0., 1.)
            weight = u.square() * (3. - 2.*u)
            # Preserve entry latch without a binary jump at either band endpoint.
            weight = torch.maximum(self.entry_weight[rows, edges], weight)
            self.entry_weight[rows, edges] = torch.where(released, 0., weight)
            armed = (weight > 0) & ~released
        self.released[rows, edges] = released
        self.armed[rows, edges] = armed
        return armed

    def penalty(self, roots, root_velocity, objects, sizes, goals, raw_success):
        armed = self._observe(roots, objects, sizes, goals, raw_success)
        penalty = torch.zeros(roots.shape[:2], dtype=roots.dtype, device=roots.device)
        weight = self.entry_weight[self.rows, self.edges] if self.soft_entry_width else armed
        penalty[self.rows, self.owners] = (self.coefficient * weight
            * root_velocity[self.rows, self.owners, :3].norm(dim=-1))
        if self.target_velocity_config is not None:
            from env.tasks.multi_agent.before_target_velocity import target_velocity_penalty
            # SIT/CLIMB track root velocity; optional ON_TOP tracking is applied below.
            selected = ~self.top
            r, e, o = self.rows[selected], self.edges[selected], self.owners[selected]
            cost, target, fade = target_velocity_penalty(
                roots[r, o], root_velocity[r, o], goals[r, self.goal[selected]],
                objects[r, self.support[selected]], sizes[r, self.support[selected]],
                self.released[r, e], self.margin, self.coefficient, self.target_velocity_config)
            penalty[r, o] = cost
            self.target_velocity = torch.zeros((*roots.shape[:2], 2), device=roots.device, dtype=roots.dtype)
            self.velocity_weight = torch.zeros(roots.shape[:2], device=roots.device, dtype=roots.dtype)
            self.target_velocity[r, o] = target
            self.velocity_weight[r, o] = fade
            if 'ontop' in self.target_velocity_config:
                selected = self.top
                r, e, o = self.rows[selected], self.edges[selected], self.owners[selected]
                self.picked_up[r, e] |= raw_success[r, self.holding_edge[selected]]
                payload = self.source[selected] - self.num_agents
                top_config = dict(self.target_velocity_config,
                                  goal_buffer=self.target_velocity_config['ontop']['goal_buffer'])
                cost, target, fade = target_velocity_penalty(
                    objects[r, payload, :3], objects[r, payload, 7:10],
                    goals[r, self.goal[selected]], objects[r, self.support[selected]],
                    sizes[r, self.support[selected]], self.released[r, e],
                    self.margin, self.coefficient, top_config)
                active = self.picked_up[r, e]
                penalty[r, o] = cost * active
                self.target_velocity[r, o] = target * active[:, None]
                self.velocity_weight[r, o] = fade * active
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
