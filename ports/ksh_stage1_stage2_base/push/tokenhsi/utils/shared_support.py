import math
from copy import deepcopy

import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import select_graph

SAMPLER = 'shared_support_groups'
PRESETS = ('random_scenario', 'all_solo', 'all_shared', 'sit', 'climb',
           'holding_at', 'holding_ontop', 'shared_sit', 'shared_climb', 'shared_ontop')


def capacities(env):
    m = env['numAgents']
    spec = env['sharedSupport']
    if env.get('multiTaskPush'):
        if env['numObjects'] != 3*m:
            raise ValueError('Five-task Stage2 requires source/PUSH/support banks: 3*M objects')
        return env['numObjects'], env['numObjects'], [1, 1]
    axes = [math.ceil(round(hi / spec['region_max_size'], 8))
            for lo, hi in spec['size_ranges'][:2]]
    minimum = 2*m + (m//2)*max(math.prod(axes)-2, 0)
    objects = env['numObjects']
    if objects < minimum:
        raise ValueError(f'Shared support requires at least {minimum} object tokens; got {objects}')
    return objects, max(4*m, objects) + m//2, axes


def pack_tokens(parents, valid, regions, num_objects, num_physical, num_agents):
    n, candidates = parents.shape
    used = torch.zeros(n, num_physical, dtype=torch.long, device=parents.device)
    used.scatter_add_(1, parents, valid.long())
    ordinary = torch.cat((torch.arange(4*num_agents, device=parents.device),
        torch.arange(4*num_agents+num_agents//2, num_physical, device=parents.device)))
    ordinary = ordinary[torch.rand(n, len(ordinary), device=parents.device).argsort(-1)]
    available = used.gather(1, ordinary) == 0
    all_parents = torch.cat((parents, ordinary), -1)
    all_valid = torch.cat((valid, available), -1)
    all_regions = torch.cat((regions, torch.full_like(ordinary, -1)), -1)
    order = all_valid.long().argsort(dim=-1, descending=True, stable=True)
    inverse = order.argsort(-1)
    selected = order[:, :num_objects]
    return (all_parents.gather(1, selected), all_regions.gather(1, selected),
            inverse[:, :candidates], valid.sum(-1))


def validate_env(env):
    from utils.unified_training import validate_unified_env
    spec = env['sharedSupport']
    capacities(env)
    if spec['target_selection'] != 'nearest_xy_per_task':
        raise ValueError('Shared support requires nearest XY target selection')
    for key in ('spawn_distance_sit', 'spawn_distance_other'):
        lo, hi = spec[key]
        if lo <= 0 or hi < lo:
            raise ValueError('Invalid shared spawn distance')
    if spec['success_region'] not in ('physical', 'selected_region'):
        raise ValueError('Unknown shared support success region')
    if not 0 <= spec['coupled_probability'] <= 1:
        raise ValueError('Invalid coupled probability')
    if spec['region_max_size'] != .8 or any(lo < .4 or hi < lo for lo, hi in spec['size_ranges'][:2]):
        raise ValueError('Shared XY must be >= .4 m; region maximum is .8 m')
    if spec['size_ranges'][2] != [.35, .5]:
        raise ValueError('Shared support height must be .35-.5 m')
    if any(weights != [1.] + [0.]*7 for weights in env['templateRsi'].values()):
        raise ValueError('Shared support experiments require loco-only initialization')
    if env['eval']['skill'] != ['loco'] or env['eval']['skillInitProb'] != [1.]:
        raise ValueError('Shared support evaluation requires loco-only initialization')
    probabilities = spec['shared_task_probabilities']
    if len(probabilities) != 3 or min(probabilities) < 0 or abs(sum(probabilities)-1) > 1e-6:
        raise ValueError('Expected probabilities for shared SIT/CLIMB/ON_TOP')
    base = deepcopy(env)
    base.pop('sharedSupport')
    base['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
    base['relationGraph']['edge_capacity'] = 4
    base['templateRsi'] = {
        'HOLDING': [.5, 0., 0., 0., 0., .5, 0., 0.],
        'SIT': [.5, .5, 0., 0., 0., 0., 0., 0.],
        'CLIMB': [.5, 0., .5, 0., 0., 0., 0., 0.],
        'HOLDING_AT': [.5, 0., 0., 0., 0., .1, .3, .1],
        'HOLDING_ON_TOP': [.5, 0., 0., 0., 0., .1, .3, .1]}
    validate_unified_env(base)


def compile_graph(spec, m, o, device=None):
    return select_graph(compose_canonical_graph(torch.full((1, m), 3, device=device,
                                                          dtype=torch.long), num_objects=o), 0)


def sample_groups(n, m, probability, device, preset='random_scenario'):
    order = torch.rand(n, m, device=device).argsort(-1)
    groups = torch.full_like(order, -1)
    cursor = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    rank = torch.arange(m, device=device)[None]
    for _ in range(m):
        remaining = m-cursor
        coupled = (remaining >= 2) & (torch.rand(n, device=device) < probability)
        if preset == 'all_solo' or preset in ('sit', 'climb', 'holding_at', 'holding_ontop'):
            coupled[:] = False
        size = torch.where(coupled, 2 + (torch.rand(n, device=device)*(remaining-1).clamp_min(0)).long(), 1)
        if preset == 'all_shared' or preset.startswith('shared_'):
            size = remaining
        active = remaining > 0
        leader = order[rows, cursor.clamp_max(m-1)]
        chosen = active[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+size)[:, None])
        ordered = groups.gather(1, order)
        ordered = torch.where(chosen, leader[:, None], ordered)
        groups.scatter_(1, order, ordered)
        cursor += size*active
    shared = (groups[:, :, None] == groups[:, None, :]).sum(-1) > 1
    return groups, shared


def sample_templates(n, m, spec, shared, device, preset):
    solo = torch.tensor([.1, .25, .325, .325], device=device)
    templates = torch.multinomial(solo, n*m, replacement=True).reshape(n, m)+1
    choices = torch.tensor([1, 2, 4], device=device)
    probs = torch.tensor(spec['shared_task_probabilities'], device=device)
    joint = choices[torch.multinomial(probs, n*m, replacement=True).reshape(n, m)]
    templates = torch.where(shared, joint, templates)
    fixed = {'sit': 1, 'climb': 2, 'holding_at': 3, 'holding_ontop': 4,
             'shared_sit': 1, 'shared_climb': 2, 'shared_ontop': 4}
    if preset in fixed:
        templates[:] = fixed[preset]
    return templates


def partition(sizes, axes, maximum=.8):
    counts = torch.ceil((sizes[..., :2]/maximum-1e-6)).long().clamp_min(1)
    ix, iy = torch.meshgrid(torch.arange(axes[0], device=sizes.device),
                            torch.arange(axes[1], device=sizes.device), indexing='ij')
    cells = torch.stack((ix.flatten(), iy.flatten()), -1)
    valid = (cells < counts[..., None, :]).all(-1)
    cell_sizes = sizes[..., None, :].expand(*sizes.shape[:-1], len(cells), 3).clone()
    cell_sizes[..., :2] /= counts[..., None, :]
    offsets = torch.zeros_like(cell_sizes)
    offsets[..., :2] = (cells+.5)*cell_sizes[..., :2]-sizes[..., None, :2]/2
    return offsets, cell_sizes, valid


def rotate(q, v):
    cross = 2*torch.cross(q[..., :3].expand_as(v), v, dim=-1)
    return v+q[..., 3:]*cross+torch.cross(q[..., :3].expand_as(v), cross, dim=-1)


def region_states(states, offsets):
    result = states[..., None, :].expand(*offsets.shape[:-1], 13).clone()
    delta = rotate(states[..., None, 3:7], offsets)
    result[..., :3] += delta
    result[..., 7:10] += torch.cross(result[..., 10:13], delta, dim=-1)
    return result


def checkpoint_contract(env):
    spec = dict(env['sharedSupport'])
    spec.pop('size_ranges')
    return dict(version=2, shared_support=spec, template_rsi=env['templateRsi'])
