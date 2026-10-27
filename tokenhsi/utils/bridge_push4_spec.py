from copy import deepcopy
import math

import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.stage1_push import PUSH


def validate_env(env):
    from utils.before_spec import validate_env as validate_before
    b = env['bridge']
    if (env['numAgents'], env['numObjects']) != (4, 8):
        raise ValueError('Bridge PUSH4 requires four humans and eight object tokens')
    if b['clearance'] != .05 or b['push']['box']['size_range'] != [1.2, 1.7]:
        raise ValueError('Bridge PUSH4 preserves cube width and 5cm clearance on each side')
    if b['task_probabilities'] != {'SIT': 1/3, 'CLIMB': 1/3, 'CARRY': 1/3}:
        raise ValueError('Bridge PUSH4 crossing tasks require equal SIT/CLIMB/CARRY')
    if b['independent_probabilities'] != [0.25]*4:
        raise ValueError('Independent tasks require equal SIT/CLIMB/CARRY/PUSH')
    if b['push']['rsi_probability'] != 1 or env['box']['reset']['platformMode'] != 'ground_only':
        raise ValueError('Bridge PUSH4 requires raw PUSH RSI and no artificial supports')
    mixture = b.get('start_mixture')
    if mixture is not None:
        low, high = mixture['near_distance']
        probability = mixture['near_probability']
        aligned = mixture.get('near_aligned_probability', 0.)
        if (not all(math.isfinite(x) for x in (low, high, probability, aligned))
                or not 0 <= probability <= 1 or not 0 <= aligned <= 1 or not 0 < low <= high):
            raise ValueError('Bridge start mixture requires a valid probability and positive distance range')
    base = deepcopy(env)
    base.pop('bridge')
    validate_before(base)


def sample_roles(count, device):
    order = torch.rand(count, 4, device=device).argsort(-1)
    rows = torch.arange(count, device=device)
    tasks = torch.randint(1, 5, (count, 4), device=device)
    tasks[rows, order[:, 0]] = 4
    tasks[rows, order[:, 1]] = torch.randint(1, 4, (count,), device=device)
    # The two independent tasks occupy different sites, avoiding crowded small stages.
    sites = torch.rand(count, 3, device=device).argsort(-1)
    surface = torch.zeros_like(tasks)
    surface[rows, order[:, 1]] = 2
    surface[rows[:, None], order[:, 2:]] = sites[:, :2]
    start_surface = surface.clone()
    start_surface[rows, order[:, 1]] = 1
    return order, tasks, surface, start_surface


def episode_graph(tasks, roles, num_objects):
    pushing = tasks == 4
    graph = compose_canonical_graph(tasks.clamp_max(3), num_objects=num_objects)
    is_push = pushing.gather(1, graph.edge_owner) & graph.edge_valid
    graph.edge_relation[(graph.edge_relation == 6) & is_push] = PUSH
    rows = torch.arange(len(tasks), device=tasks.device)
    pusher, crosser = roles[:, 0], roles[:, 1]
    leader = ((graph.edge_owner == pusher[:, None]) & (graph.edge_relation == 7) & graph.edge_valid).long().argmax(-1)
    follower = ((graph.edge_owner == crosser[:, None]) & graph.required_goal).long().argmax(-1)
    graph.prereq_mask[rows, follower, leader] = True
    predecessor = torch.full_like(tasks, -1)
    predecessor[rows, crosser] = pusher
    return graph, predecessor
