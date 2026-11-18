
import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import select_graph

SAMPLER = 'stage2_before_groups'
PRESETS = ('random_scenario', 'before_sit', 'before_climb', 'before_stack',
           'one_pair_sit', 'one_pair_climb', 'one_pair_stack', 'all_solo',
           'sit', 'climb', 'holding_at', 'holding_ontop')


def validate_env(env):
    if env.get("jointPush") is not None:
        from utils.before_push_spec import validate_env as validate_push
        return validate_push(env)
    from utils.experiments import BEFORE_NAMES, native_config
    from utils.unified_training import validate_common_env
    if env['numAgents'] < 1 or env['numObjects'] < 2*env['numAgents']:
        raise ValueError('BEFORE object aliases require numObjects >= 2*numAgents')
    validate_common_env(env)
    candidates = [native_config(name)['env'] for name in BEFORE_NAMES]
    if not any(checkpoint_contract(env) == checkpoint_contract(native) for native in candidates):
        raise ValueError('Unsupported BEFORE configuration contract')
    if not any(env['relationGraph'] == native['relationGraph'] for native in candidates):
        raise ValueError('Unsupported BEFORE graph contract')
    if any(row != [1.] + [0.]*7 for row in env['templateRsi'].values()):
        raise ValueError('BEFORE base initialization must be loco')
    if env['eval']['skill'] != ['loco'] or env['eval']['skillInitProb'] != [1.]:
        raise ValueError('BEFORE evaluation requires loco; follower RSI has a separate option')


def compile_graph(spec, m, o, device=None):
    return select_graph(compose_canonical_graph(torch.full((1, m), 3,
        device=device, dtype=torch.long), num_objects=o), 0)


def eligible_families(sizes, ranges):
    masks = []
    for name in ('SIT', 'CLIMB'):
        bounds = sizes.new_tensor(ranges[name])
        masks.append(((sizes >= bounds[:, 0]-1e-6) & (sizes <= bounds[:, 1]+1e-6)).all(-1))
    return torch.stack((*masks, torch.ones_like(masks[0])), -1)


def sample_roles(sizes, ranges, preset, solo_probabilities=None, pairing_probability=.8):
    if preset not in PRESETS:
        raise ValueError('Unknown BEFORE preset: '+preset)
    n, m = sizes.shape[:2]
    device = sizes.device
    grouping = 'two_pairs_at' if preset.startswith('before_') else (
        'one_pair_at' if preset.startswith('one_pair_') else preset)
    groups = sample_groups(n, device, grouping, num_agents=m, pairing_probability=pairing_probability)
    paired = (groups[:, :, None] == groups[:, None]).sum(-1) == 2
    templates = torch.multinomial(sizes.new_tensor([.1, .25, .325, .325] if solo_probabilities is None else solo_probabilities),
        n*m, replacement=True).reshape(n, m)+1
    fixed = {'sit': 1, 'climb': 2, 'holding_at': 3, 'holding_ontop': 4}
    if preset in fixed:
        templates[:] = fixed[preset]
    predecessor = torch.full_like(groups, -1)
    allowed = eligible_families(sizes, ranges)
    for leader in range(m):
        rows = (paired[:, leader] & (groups[:, leader] == leader)).nonzero().flatten()
        if not len(rows):
            continue
        partner = ((groups[rows] == leader) & (torch.arange(m, device=device) != leader)).long().argmax(-1)
        candidates = torch.stack((torch.full_like(partner, leader), partner), -1)
        available = allowed[rows[:, None], candidates]
        weights = available.any(1).float()
        if preset != 'random_scenario':
            family = ('sit', 'climb', 'stack').index(preset.rsplit('_', 1)[-1])
            weights *= torch.nn.functional.one_hot(torch.tensor(family, device=device), 3)
        if (weights.sum(-1) == 0).any():
            raise ValueError('No size-compatible BEFORE role for preset '+preset)
        family = torch.multinomial(weights, 1).flatten()
        choices = available.gather(2, family[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        role = torch.multinomial(choices.float(), 1)
        first = candidates.gather(1, role).flatten()
        second = candidates.gather(1, 1-role).flatten()
        templates[rows, first] = 3
        templates[rows, second] = torch.tensor([1, 2, 4], device=device)[family]
        predecessor[rows, second] = first
    return predecessor, templates


def task_objects(predecessor, templates):
    follower = predecessor >= 0
    return torch.cat((~follower | (templates == 4), ~follower & (templates == 4)), -1)


def sample_graph(sizes, ranges, preset='random_scenario', num_objects=None):
    n, m = sizes.shape[:2]
    o = 2*m if num_objects is None else num_objects
    if m < 1 or o < m:
        raise ValueError('BEFORE requires numAgents >= 1 and numObjects >= numAgents')
    if preset == 'holding_ontop' and o < 2*m:
        raise ValueError('holding_ontop requires at least two objects per human')
    if preset.startswith(('before_', 'one_pair_')) and m < 2:
        raise ValueError('BEFORE pair presets require at least two humans')
    predecessor = torch.empty(n, m, device=sizes.device, dtype=torch.long)
    templates = torch.empty_like(predecessor)
    pending = torch.arange(n, device=sizes.device)
    for _ in range(1024):
        pre, sampled = sample_roles(sizes[pending], ranges, preset)
        fits = task_objects(pre, sampled).sum(-1) <= o
        predecessor[pending[fits]], templates[pending[fits]] = pre[fits], sampled[fits]
        pending = pending[~fits]
        if not len(pending):
            break
    else:
        raise RuntimeError('BEFORE task graph exceeds object budget after 1024 attempts; increase numObjects')
    graph = compose_canonical_graph(templates, num_objects=2*m)
    rows, follower = (predecessor >= 0).nonzero(as_tuple=True)
    edge = ((graph.edge_owner[rows] == follower[:, None]) & graph.required_goal[rows]).long().argmax(-1)
    first_edge = ((graph.edge_owner[rows] == predecessor[rows, follower, None])
                  & (graph.edge_relation[rows] == 7) & graph.edge_valid[rows]).long().argmax(-1)
    graph.edge_dst[rows, edge] = m+predecessor[rows, follower]
    graph.prereq_mask[rows, edge, first_edge] = True
    return graph, predecessor, templates


def checkpoint_contract(env):
    contract = dict(version=1, before=env['before'], template_rsi=env['templateRsi'],
                    sizes=env['box']['build']['taskSizeRanges'], reset=env['box']['reset'])
    for key in ('actor_human_bias', 'critic_human_bias'):
        mode = env['relationGraph'].get(key, 'all')
        if mode != 'all':
            contract[key] = mode
    if env['relationGraph'].get('before_object_alias') is not None:
        contract['before_object_alias'] = env['relationGraph']['before_object_alias']
    if env['before'].get('follower_pickup_carry_rsi') is not None:
        contract['size_aware_rsi'] = env['sizeAwareRsi']
    if env.get("jointPush") is not None:
        from utils.before_push_spec import push_contract
        contract["solo_push"] = push_contract(env)
    return contract


def check_checkpoint(weights, env):
    from utils.experiments import check_before_checkpoint
    check_before_checkpoint(weights, env)


def pack_objects(parents, valid, num_objects=None, num_physical=None, excluded_distractor_range=None):
    o = parents.shape[1] if num_objects is None else num_objects
    physical = max(2*parents.shape[1], o) if num_physical is None else num_physical
    if (valid.sum(-1) > o).any():
        raise ValueError('Task objects exceed the token budget')
    used = torch.zeros(len(parents), physical, device=parents.device, dtype=torch.long)
    used.scatter_add_(1, parents, valid.long())
    ordinary = torch.rand(len(parents), physical, device=parents.device).argsort(-1)
    available = used.gather(1, ordinary) == 0
    if excluded_distractor_range is not None:
        lo, hi = excluded_distractor_range
        available &= (ordinary < lo) | (ordinary >= hi)
    all_parents = torch.cat((parents, ordinary), -1)
    order = torch.cat((valid, available), -1).long().argsort(dim=-1, descending=True, stable=True)
    return all_parents.gather(1, order[:, :o]), order.argsort(-1)[:, :parents.shape[1]], valid.sum(-1)


def sample_groups(n, device, preset='random_scenario', num_agents=4, pairing_probability=.8):
    order = torch.rand(n, num_agents, device=device).argsort(-1)
    remaining = torch.ones(n, num_agents, dtype=torch.bool, device=device)
    groups = torch.full((n, num_agents), -1, device=device, dtype=torch.long)
    rows = torch.arange(n, device=device)
    count = torch.zeros(n, device=device, dtype=torch.long)
    for _ in range(num_agents):
        active = remaining.any(-1)
        first = order.gather(1, remaining.gather(1, order).long().argmax(-1)[:, None]).flatten()
        remaining[rows, first] = False
        paired = active & remaining.any(-1) & (torch.rand(n, device=device) < pairing_probability)
        if preset.startswith('two_pairs'):
            paired = active & remaining.any(-1)
        elif preset.startswith('one_pair'):
            paired = active & remaining.any(-1) & (count == 0)
        elif preset != 'random_scenario':
            paired[:] = False
        partner = torch.rand(n, num_agents, device=device).masked_fill(~remaining, -1).argmax(-1)
        leader = torch.where(paired, torch.minimum(first, partner), first)
        groups[rows[active], first[active]] = leader[active]
        groups[rows[paired], partner[paired]] = leader[paired]
        remaining[rows[paired], partner[paired]] = False
        count += paired
    return groups
