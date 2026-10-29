from copy import deepcopy

import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import select_graph
from utils.stage2_spec import sample_groups

SAMPLER = 'stage2_before_groups'
PRESETS = ('random_scenario', 'before_sit', 'before_climb', 'before_stack',
           'one_pair_sit', 'one_pair_climb', 'one_pair_stack', 'all_solo',
           'sit', 'climb', 'holding_at', 'holding_ontop')
SPEC = dict(pairing_probability=.8, follower_weights=[1., 1., 1.],
            gate='current_at_success', progress='always', holding='always',
            early_penalty=0., initialization='loco')


def validate_env(env):
    from utils.unified_training import validate_unified_env
    from utils.before_sampling import DISTANCE_MIXTURE, THREE_WAY_DISTANCE_MIXTURE, MIX226_DISTANCE_MIXTURE
    if env['numAgents'] < 1 or env['numObjects'] < env['numAgents']:
        raise ValueError('BEFORE requires numAgents >= 1 and numObjects >= numAgents')
    before = dict(env['before'])
    if env['relationGraph'].get('actor_human_bias', 'all') not in ('all', 'before_only', 'before_only_mask_none'):
        raise ValueError('Unknown BEFORE actor_human_bias')
    if env['relationGraph'].get('critic_human_bias', 'all') not in ('all', 'before_only', 'before_only_mask_none'):
        raise ValueError('Unknown BEFORE critic_human_bias')
    if env['relationGraph'].get('actor_head_adapter') not in (None, 'tokenhsi_internal_v1'):
        raise ValueError('Unknown BEFORE actor_head_adapter')
    from utils.before_object_alias import MODE as OBJECT_ALIAS_MODE
    alias = env['relationGraph'].get('before_object_alias')
    if alias not in (None, OBJECT_ALIAS_MODE):
        raise ValueError('Unknown BEFORE before_object_alias')
    if alias is not None and env['numObjects'] < 2*env['numAgents']:
        raise ValueError('BEFORE object aliases require numObjects >= 2*numAgents')
    from utils.before_full_rsi import SPEC as FULL_SPEC, TEMPLATE_RSI
    full = before.pop('full_rsi', None)
    if full is not None and (full != FULL_SPEC or env['templateRsi'] != TEMPLATE_RSI
            or env['sizeAwareRsi']['lateProbability'] != .7
            or env['sizeAwareRsi']['lateFraction'] != .3):
        raise ValueError('Full BEFORE RSI requires the Stage1 probabilities and late frames')
    if full is not None:
        if before.get('initialization') != 'stage1_rsi':
            raise ValueError('Full BEFORE RSI requires initialization=stage1_rsi')
        before['initialization'] = 'loco'  # Normalize only for the legacy base contract below.
    pickup_carry = before.pop('follower_pickup_carry_rsi', None)
    from utils.before_pickup_carry_rsi import SPEC as PICKUP_CARRY_SPEC
    if pickup_carry is not None and pickup_carry != PICKUP_CARRY_SPEC:
        raise ValueError('BEFORE pickup/carry RSI requires loco/pickUp/carryWith 40/30/30')
    carry_rsi = before.pop('follower_carry_rsi', None)
    if pickup_carry is not None and (full is not None or carry_rsi is not None):
        raise ValueError('BEFORE RSI initializations are mutually exclusive')
    if full is not None and carry_rsi is not None:
        raise ValueError('Full RSI and follower-only carry RSI are mutually exclusive')
    if carry_rsi is not None and carry_rsi != dict(probability=.5, skill='carryWith'):
        raise ValueError('BEFORE follower_carry_rsi requires carryWith 50/50')
    mixture = before.pop('distance_mixture', None)
    from utils.before_relative_sampling import DISTANCE_MIXTURE as RELATIVE_MIXTURE
    progress_target = before.pop('progress_target', 'current_box')
    if progress_target not in ('current_box', 'predecessor_at_goal'):
        raise ValueError('Unknown BEFORE progress_target')
    if progress_target == 'predecessor_at_goal' and (
            mixture not in (MIX226_DISTANCE_MIXTURE, RELATIVE_MIXTURE) or env['box']['reset']['platformMode'] != 'ground_only'):
        raise ValueError('BEFORE AT-goal progress requires ground mix226 or relative-sum distances')
    if mixture is not None and mixture not in (DISTANCE_MIXTURE, THREE_WAY_DISTANCE_MIXTURE, MIX226_DISTANCE_MIXTURE, RELATIVE_MIXTURE):
        raise ValueError('Unknown BEFORE distance mixture')
    if (carry_rsi is not None or pickup_carry is not None) and (progress_target != 'predecessor_at_goal' or mixture not in (MIX226_DISTANCE_MIXTURE, RELATIVE_MIXTURE)):
        raise ValueError('BEFORE follower RSI requires AT progress with mix226 or relative-sum distances')
    if full is not None and progress_target != 'predecessor_at_goal':
        raise ValueError('Full BEFORE RSI requires mix226 AT progress')
    if before != SPEC or env['relationGraph']['sampler'] != SAMPLER:
        raise ValueError('BEFORE configuration differs from the agreed contract')
    if full is None and any(row != [1.] + [0.]*7 for row in env['templateRsi'].values()):
        raise ValueError('BEFORE requires loco-only initialization')
    if env['eval']['skill'] != ['loco'] or env['eval']['skillInitProb'] != [1.]:
        raise ValueError('BEFORE evaluation requires loco only')
    base = deepcopy(env)
    base.pop('before')
    base['relationGraph'].pop('actor_human_bias', None)
    base['relationGraph'].pop('critic_human_bias', None)
    base['relationGraph'].pop('actor_head_adapter', None)
    base['relationGraph'].pop('before_object_alias', None)
    reset = base['box']['reset']
    if reset['platformMode'] == 'ground_only':
        if reset['randomHeight'] is not False or reset['randomHeightProb'] != 0.:
            raise ValueError('Ground BEFORE requires randomHeight=false and randomHeightProb=0')
        reset.update(platformMode='carry_payload_tokenhsi', randomHeight=True, randomHeightProb=.5)
    base['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
    base['relationGraph']['edge_capacity'] = 4
    for name in ('HOLDING_AT', 'HOLDING_ON_TOP'):
        base['templateRsi'][name] = [.5, 0., 0., 0., 0., .1, .3, .1]
    validate_unified_env(base)


def compile_graph(spec, m, o, device=None):
    return select_graph(compose_canonical_graph(torch.full((1, m), 3,
        device=device, dtype=torch.long), num_objects=o), 0)


def eligible_families(sizes, ranges):
    masks = []
    for name in ('SIT', 'CLIMB'):
        bounds = sizes.new_tensor(ranges[name])
        masks.append(((sizes >= bounds[:, 0]-1e-6) & (sizes <= bounds[:, 1]+1e-6)).all(-1))
    return torch.stack((*masks, torch.ones_like(masks[0])), -1)


def sample_roles(sizes, ranges, preset):
    if preset not in PRESETS:
        raise ValueError('Unknown BEFORE preset: '+preset)
    n, m = sizes.shape[:2]
    device = sizes.device
    grouping = 'two_pairs_at' if preset.startswith('before_') else (
        'one_pair_at' if preset.startswith('one_pair_') else preset)
    groups = sample_groups(n, device, grouping, num_agents=m)
    paired = (groups[:, :, None] == groups[:, None]).sum(-1) == 2
    templates = torch.multinomial(sizes.new_tensor([.1, .25, .325, .325]),
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
    # Preserve legacy contracts byte-for-byte when the option is absent/default.
    mode = env['relationGraph'].get('actor_human_bias', 'all')
    if mode != 'all':
        contract['actor_human_bias'] = mode
    critic_mode = env['relationGraph'].get('critic_human_bias', 'all')
    if critic_mode != 'all':
        contract['critic_human_bias'] = critic_mode
    if env['relationGraph'].get('actor_head_adapter') is not None:
        contract['actor_head_adapter'] = env['relationGraph']['actor_head_adapter']
    if env['relationGraph'].get('before_object_alias') is not None:
        contract['before_object_alias'] = env['relationGraph']['before_object_alias']
    if env['before'].get('full_rsi') is not None or env['before'].get('follower_pickup_carry_rsi') is not None:
        contract['size_aware_rsi'] = env['sizeAwareRsi']
    return contract


def pack_objects(parents, valid, num_objects=None):
    o = parents.shape[1] if num_objects is None else num_objects
    physical = max(2*parents.shape[1], o)
    if (valid.sum(-1) > o).any():
        raise ValueError('Task objects exceed the token budget')
    used = torch.zeros(len(parents), physical, device=parents.device, dtype=torch.long)
    used.scatter_add_(1, parents, valid.long())
    ordinary = torch.rand(len(parents), physical, device=parents.device).argsort(-1)
    available = used.gather(1, ordinary) == 0
    all_parents = torch.cat((parents, ordinary), -1)
    order = torch.cat((valid, available), -1).long().argsort(dim=-1, descending=True, stable=True)
    return all_parents.gather(1, order[:, :o]), order.argsort(-1)[:, :parents.shape[1]], valid.sum(-1)
