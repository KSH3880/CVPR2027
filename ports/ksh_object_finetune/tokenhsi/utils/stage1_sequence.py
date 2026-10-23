import math

import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.size_rsi import SIZE_RANGES, TASKS as SIZE_TASKS
from utils.task_role_spec import task_size_probabilities


def parse_task_sequences(a='', b=''):
    names = dict(sit=1, climb=2, at=3, carry_at=3, ontop=4, carry_ontop=4)
    fixed = torch.zeros(2, 2, dtype=torch.long)
    for agent, value in enumerate((a, b)):
        if not value.strip():
            continue
        tasks = [name.strip().lower() for name in value.split(',')]
        if len(tasks) != 2 or any(name not in names for name in tasks):
            raise ValueError('TASK_SEQUENCE_A/B must contain two tasks: sit, climb, at, ontop')
        fixed[:, agent] = torch.tensor([names[name] for name in tasks])
    return fixed


def sample_full_sequence(fixed, generator):
    first = torch.randint(1, 5, (2,), device=fixed.device, generator=generator)
    second = (first - 1 + torch.randint(1, 4, (2,), device=fixed.device, generator=generator)) % 4 + 1
    templates = torch.stack((first, second))
    return torch.where(fixed > 0, fixed, templates), torch.arange(2, device=fixed.device).repeat(2, 1)


def sequence_box_sizes(fixed, device):
    sizes = torch.empty(1, 4, 3, device=device)
    for agent in range(2):
        ids = fixed[:, agent].tolist() if fixed[:, agent].any() else range(1, 5)
        bounds = torch.tensor([SIZE_RANGES[SIZE_TASKS[index]] for index in ids])
        low = bounds[..., 0].amax(0)
        high = bounds[..., 1].amin(0)
        for axis in range(3):
            sizes[0, agent, axis] = torch.randint(round(float(low[axis]) * 20),
                round(float(high[axis]) * 20) + 1, (), device=device) / 20.
    for axis, (low, high) in enumerate(SIZE_RANGES['SUPPORT']):
        sizes[0, 2:, axis] = torch.randint(round(low * 20), round(high * 20) + 1,
                                         (2,), device=device) / 20.
    return sizes


def sample_sequence(sizes, generator):
    weights = task_size_probabilities(sizes[:2])
    first = torch.multinomial(weights, 1, generator=generator).flatten()
    second_weights = weights.clone()
    second_weights.scatter_(1, first[:, None], 0)
    targets = torch.arange(2, device=sizes.device).repeat(2, 1)
    for agent in range(2):
        if second_weights[agent].sum() == 0:
            fallback = task_size_probabilities(sizes[agent + 2:agent + 3])[0]
            fallback[3:] = 0
            distinct = fallback.clone()
            distinct[first[agent]] = 0
            second_weights[agent] = distinct if distinct.sum() > 0 else fallback
            targets[1, agent] = agent + 2
    second = torch.multinomial(second_weights, 1, generator=generator).flatten()
    return torch.stack((first, second)), targets


def sequence_graph(templates, targets):
    graph = compose_canonical_graph(templates[None])
    interaction = graph.edge_valid & ((graph.edge_relation == 9) | (graph.edge_relation == 10))
    graph.edge_dst[interaction] = (targets[graph.edge_owner] + 2)[interaction]
    return graph


class SequenceProgress:
    def __init__(self, mode, seconds, dt, device):
        if mode not in ('independent', 'synchronized'):
            raise ValueError('Sequence mode must be independent or synchronized')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Success hold seconds must be positive and finite')
        self.mode = mode
        self.hold_steps = math.ceil(seconds / dt)
        self.stage = torch.zeros(2, dtype=torch.long, device=device)
        self.stable = torch.zeros_like(self.stage)
        self.completed = torch.zeros(2, 2, dtype=torch.bool, device=device)

    def step(self, success):
        self.stable = torch.where(success, self.stable + 1, 0)
        ready = self.stable >= self.hold_steps
        agents = torch.arange(2, device=success.device)
        newly = ready & ~self.completed[agents, self.stage]
        self.completed[agents[newly], self.stage[newly]] = True
        advance = ready & (self.stage == 0)
        if self.mode == 'synchronized':
            advance &= ready.all()
        self.stage[advance] = 1
        self.stable[advance] = 0
        return newly, advance

    def finished(self):
        return bool(self.completed.all())
