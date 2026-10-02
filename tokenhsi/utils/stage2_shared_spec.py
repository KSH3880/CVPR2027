"""Nine shared-object pairs with immutable, role-sized four-object assets."""
import math

import torch

from utils.edge_context_spec import AT, HOLDING
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP, permute_graph

SAMPLER = 'two_agent_four_object_stage2_rescue_shared9'
PAIRS = {
    'at_climb': ('HOLDING_AT', 'CLIMB'),
    'at_sit': ('HOLDING_AT', 'SIT'),
    'at_ontop': ('HOLDING_AT', 'HOLDING_ON_TOP'),
    'ontop_ontop': ('HOLDING_ON_TOP', 'HOLDING_ON_TOP'),
    'ontop_climb': ('HOLDING_ON_TOP', 'CLIMB'),
    'ontop_sit': ('HOLDING_ON_TOP', 'SIT'),
    'climb_climb': ('CLIMB', 'CLIMB'),
    'sit_climb': ('SIT', 'CLIMB'),
    'sit_sit': ('SIT', 'SIT'),
}
ALIASES = {'place_climb': 'at_climb', 'place_sit': 'at_sit', 'place_stack': 'at_ontop'}
PRESETS = ('random_scenario', 'independent', *ALIASES, *PAIRS)
# Pool 0: independent (ordinary assets); 1: AT pairs (ordinary); 2: shared.
POOL_PROBS = (.1, .45, .45)
ROLE_RANGES = {
    'ordinary': [[.4, .6], [.4, .6], [.3, .5]],
    'support': [[.5, .65], [.65, .7], [.45, .5]],
    'payload': [[.3, .4], [.3, .4], [.4, .5]],
}


def validate_sampler(spec, m, o):
    from utils.edge_stage2_spec import STAGE2_RESCUE_SAMPLER, validate_sampler as validate_legacy
    legacy = dict(spec, sampler=STAGE2_RESCUE_SAMPLER,
                  family_probabilities=dict(place_climb=1/3, place_sit=1/3, place_stack=1/3))
    validate_legacy(legacy, m, o)
    expected = {name: (1/6 if name.startswith('at_') else 1/12) for name in PAIRS}
    values = spec.get('family_probabilities', {})
    if set(values) != set(expected) or any(
            isinstance(values[k], bool) or not isinstance(values[k], (float, int)) or
            not math.isclose(values[k], v, abs_tol=1e-12) for k, v in expected.items()):
        raise ValueError('Shared9 requires AT pairs 15% each, other pairs 7.5% each, independent 10%')


def validate_env(env):
    validate_sampler(env['relationGraph'], env['numAgents'], env['numObjects'])
    sizes = env['box']['build'].get('stage2RoleSizes')
    if sizes != dict(step=.05, **ROLE_RANGES):
        raise ValueError('Shared9 requires its ordinary/support/payload 5cm role-size grids')
    if tuple(env['skill']) != ('loco', 'sit', 'climb', 'climbNoRSI', 'omomo', 'pickUp', 'carryWith', 'putDown'):
        raise ValueError('Shared9 must retain all eight AMP expert skills')
    loco = [1., 0., 0., 0., 0., 0., 0., 0.]
    from utils.edge_scenario_spec import CLIMB_TEMPLATES
    if env['skillInitProb'] != loco or any(
            tuple(env.get(key, {})) != CLIMB_TEMPLATES or
            any(row != loco for row in env[key].values())
            for key in ('templateRsi', 'independentTemplateRsi')):
        raise ValueError('Shared9 initializes every task from loco; AMP experts remain separate')
    if (env['box']['reset'].get('initialBodyClearance') != .08 or
            env['box']['reset'].get('randomHeight') is not False):
        raise ValueError('Shared9 requires ground spawning and 8cm initial body-sphere clearance')


def scene_pools(n, device='cpu', preset='random_scenario', generator=None):
    """Largest-remainder quotas, shuffled once before immutable asset creation."""
    preset = ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError('Unknown Shared9 preset: ' + preset)
    if preset != 'random_scenario':
        pool = 0 if preset == 'independent' else 1 if preset.startswith('at_') else 2
        return torch.full((n,), pool, device=device, dtype=torch.long)
    if n < 10:
        return torch.multinomial(torch.tensor(POOL_PROBS, device=device), n,
                                 replacement=True, generator=generator)
    exact = torch.tensor(POOL_PROBS, dtype=torch.float64) * n
    counts = exact.floor().long()
    remainder = n - int(counts.sum())
    counts[(exact - counts).argsort(descending=True)[:remainder]] += 1
    pools = torch.repeat_interleave(torch.arange(3), counts).to(device)
    return pools[torch.randperm(n, device=device, generator=generator)]


def sample_role_sizes(pools, sizes, generator=None):
    """Physical slot 0 support, 1/2 payload, 3 ordinary in shared pools only."""
    result = torch.empty(len(pools), 4, 3, device=pools.device)
    for role, columns in (('ordinary', range(4)), ('support', (0,)), ('payload', (1, 2))):
        rows = torch.arange(len(pools), device=pools.device) if role == 'ordinary' else (pools == 2).nonzero().flatten()
        if not len(rows):
            continue
        for col in columns:
            for axis, (low, high) in enumerate(sizes[role]):
                count = round((high - low) / sizes['step']) + 1
                result[rows, col, axis] = low + sizes['step'] * torch.randint(
                    count, (len(rows),), device=pools.device, generator=generator)
    return result


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None, pools=None):
    from utils.edge_stage2_spec import _empty_graph, _put, validate_graph
    from utils.edge_scenario_spec import UNIFIED_SAMPLER, _rows, sample_graph as independent
    validate_sampler(spec, 2, 4)
    preset = ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError('Unknown Shared9 preset: ' + preset)
    if pools is None:
        pools = scene_pools(n, device, preset, generator)
    if pools.shape != (n,) or ((pools < 0) | (pools > 2)).any():
        raise ValueError('Invalid Shared9 scene pools')
    graph = _empty_graph(n, 2, 4, 4, device)
    independent_spec = dict(mode='edge_composition', sampler=UNIFIED_SAMPLER,
        semantic_only=True, edge_capacity=4, max_edges_per_agent=2,
        template_probabilities=spec['independent_template_probabilities'],
        random_binding={'objects': 'canonical', 'goals': 'canonical'},
        shuffle_edge_order=False, shuffle_token_order=True)
    for row in range(n):
        pool = int(pools[row])
        if preset != 'random_scenario':
            expected_pool = 0 if preset == 'independent' else 1 if preset.startswith('at_') else 2
            if pool != expected_pool:
                raise ValueError('Preset does not match immutable Shared9 asset pool')
        if pool == 0:
            old = independent(1, independent_spec, device, generator=generator)
            for key in ('edge_src', 'edge_dst', 'edge_relation', 'edge_owner', 'edge_valid', 'required_goal'):
                getattr(graph, key)[row] = getattr(old, key)[0]
            continue
        choices = tuple(PAIRS)[:3] if pool == 1 else tuple(PAIRS)[3:]
        family = choices[int(torch.randint(len(choices), (), device=device, generator=generator))] if preset == 'random_scenario' else preset
        owners = torch.randperm(2, device=device, generator=generator).tolist() if preset == 'random_scenario' else ([1, 0] if role_swap else [0, 1])
        objects = torch.randperm(4, device=device, generator=generator).tolist()
        base, children = objects[0], iter(objects[1:])
        goal = int(torch.randint(2, (), device=device, generator=generator))
        offset = 0
        for owner, template in zip(owners, PAIRS[family]):
            binding = (base, goal) if template == 'HOLDING_AT' else (next(children), base) if template == 'HOLDING_ON_TOP' else (base,)
            for src, dst, relation, required in _rows(owner, template, binding, 2, 4):
                _put(graph, row, offset, owner, src, dst, relation, required)
                offset += 1
    if spec['shuffle_edge_order']:
        graph = permute_graph(graph, torch.rand(n, 4, device=device, generator=generator).argsort(-1))
    validate_graph(graph, allow_shared_ontop=True)
    return graph


def box_order(graph, pools, generator=None):
    """Logical-to-physical permutation; shared role constraints never depend on owners."""
    order = torch.rand(len(pools), 4, device=pools.device, generator=generator).argsort(-1)
    for row in (pools == 2).nonzero().flatten().tolist():
        valid = graph.edge_valid[row]
        relevant = valid & ((graph.edge_relation[row] == ON_TOP) |
                           (graph.edge_relation[row] == SIT) | (graph.edge_relation[row] == CLIMB))
        targets = graph.edge_dst[row, relevant].unique()
        if len(targets) != 1:
            raise ValueError('Shared9 must bind both tasks to one shared support')
        base = int(targets[0]) - 2
        children = (graph.edge_src[row, valid & (graph.edge_relation[row] == ON_TOP)] - 2).tolist()
        payloads = (1 + torch.randperm(2, device=pools.device, generator=generator)).tolist()
        fixed = {base: 0, **dict(zip(children, payloads))}
        remaining = [i for i in order[row].tolist() if i not in fixed.values()]
        for logical in range(4):
            order[row, logical] = fixed[logical] if logical in fixed else remaining.pop()
    return order


def initial_body_collisions(bodies, boxes, sizes, clearance=.08):
    """Conservative sphere proxies at FK body centers, against every box/human."""
    relative = bodies[:, :, :, None, :] - boxes[:, None, None, :, :3]
    rotation = boxes[:, None, None, :, 3:7].expand(*relative.shape[:-1], 4)
    cross = 2 * torch.cross(rotation[..., :3], relative, dim=-1)
    local = relative - rotation[..., 3:4] * cross + torch.cross(rotation[..., :3], cross, dim=-1)
    outside = (local.abs() - sizes[:, None, None] / 2).clamp_min(0).norm(dim=-1)
    box_bad = (outside < clearance).flatten(1).any(-1)
    human_bad = torch.zeros(len(bodies), device=bodies.device, dtype=torch.bool)
    for first in range(bodies.shape[1]):
        for second in range(first + 1, bodies.shape[1]):
            human_bad |= ((bodies[:, first, :, None] - bodies[:, second, None]).norm(dim=-1)
                          < 2 * clearance).flatten(1).any(-1)
    return box_bad | human_bad | (bodies[..., 2].amin((1, 2)) < -.02)
