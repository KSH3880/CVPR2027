from copy import deepcopy

import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import permute_graph


SAMPLER = 'stage2_random_groups'
PRESETS = ('random_scenario', 'two_pairs_at', 'two_pairs_ontop', 'one_pair_at',
           'one_pair_ontop', 'all_solo', 'sit', 'climb', 'carry_at', 'carry_ontop', 'holding_at', 'holding_ontop')


def sample_groups(n, device, preset='random_scenario', num_agents=4):
    order = torch.rand(n, num_agents, device=device).argsort(-1)
    remaining = torch.ones(n, num_agents, dtype=torch.bool, device=device)
    groups = torch.full((n, num_agents), -1, device=device, dtype=torch.long)
    rows = torch.arange(n, device=device)
    count = torch.zeros(n, device=device, dtype=torch.long)
    for _ in range(num_agents):
        active = remaining.any(-1)
        first = order.gather(1, remaining.gather(1, order).long().argmax(-1)[:, None]).flatten()
        remaining[rows, first] = False
        paired = active & remaining.any(-1) & (torch.rand(n, device=device) < .8)
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


def sample_graph(probabilities, spec, preset='random_scenario'):
    if preset not in PRESETS or spec.get('sampler') != SAMPLER:
        raise ValueError('Unsupported Stage2 group preset')
    n, m = probabilities.shape[:2]
    if m != 4:
        raise ValueError('Stage2 requires four humans')
    device = probabilities.device
    groups = sample_groups(n, device, preset)
    joint = (groups[:, :, None] == groups[:, None, :]).sum(-1) == 2
    templates = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(n, m)
    joint_templates = (3 + (torch.rand(n, m, device=device) < .5).long()).gather(1, groups)
    if preset.endswith('_at'):
        joint_templates[:] = 3
    elif preset.endswith('_ontop'):
        joint_templates[:] = 4
    single = {'sit': 1, 'climb': 2, 'carry_at': 3, 'carry_ontop': 4, 'holding_at': 3, 'holding_ontop': 4}
    if preset in single:
        templates[:] = single[preset]
    templates = torch.where(joint, joint_templates, templates)
    graph = compose_canonical_graph(templates, num_objects=8)
    owner = graph.edge_owner
    leader = groups.gather(1, owner)
    shared = joint.gather(1, owner)
    holding = graph.edge_relation == 6
    placement = (graph.edge_relation == 7) | (graph.edge_relation == 8)
    graph.edge_dst.copy_(torch.where(shared & holding, m + leader, graph.edge_dst))
    graph.edge_src.copy_(torch.where(shared & placement, m + leader, graph.edge_src))
    target = torch.where(graph.edge_relation == 8, 2*m + leader, 3*m + leader)
    graph.edge_dst.copy_(torch.where(shared & placement, target, graph.edge_dst))
    if spec.get('shuffle_edge_order', False):
        graph = permute_graph(graph, torch.rand(n, 8, device=device).argsort(-1))
    return graph, groups, joint


def physical_mapping(groups, joint):
    n, m = groups.shape
    ids = torch.arange(m, device=groups.device)[None].expand(n, -1)
    leaders = joint & (groups == ids)
    index = (leaders.long().cumsum(-1) - 1).gather(1, groups)
    source = torch.where(joint, torch.where(groups == ids, 8 + index, 10 + index), ids)
    return torch.cat((source, ids + m), -1)


def validate_env(env):
    from utils.unified_training import validate_unified_env
    if (env['numAgents'], env['numObjects']) != (4, 8):
        raise ValueError('Stage2 supports four humans and eight scene objects')
    base = deepcopy(env)
    base.pop('stage2')
    base['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
    base['relationGraph']['edge_capacity'] = 4
    if base['templateRsi']['HOLDING_ON_TOP'] != [.5, 0., 0., 0., 0., .1, .4, 0.]:
        raise ValueError('Stage2 ON_TOP RSI requires 50/10/40/0')
    base['templateRsi']['HOLDING_ON_TOP'] = list(base['templateRsi']['HOLDING_AT'])
    validate_unified_env(base)
    expected = dict(pairing_probability=.8, ontop_probability=.5,
                    joint_size_ranges=[[.2, .6], [.4, 1.2], [.2, .6]],
                    anchor_inset=.15, rsi_directory='joint_carry/stage2_size_rsi/inset15_single',
                    amp_motion_file='joint_carry/teamhoi_retarget_posture_all/amp_motions.yaml')
    if env['stage2'] != expected or env['relationGraph']['sampler'] != SAMPLER:
        raise ValueError('Stage2 configuration differs from the agreed contract')


def check_checkpoint(weights, env):
    if "before" in env:
        from utils.before_spec import checkpoint_contract
        if weights.get("before_contract") != checkpoint_contract(env):
            raise ValueError("BEFORE checkpoint contract differs")
        return
    if "sharedSupport" in env:
        from utils.shared_support import checkpoint_contract
        if weights.get("shared_support_contract") != checkpoint_contract(env):
            raise ValueError("Shared support checkpoint contract differs")
        return
    from joint_carry.artifacts import contract_digest
    if weights.get('stage2_rsi_contract') != contract_digest():
        raise ValueError('Stage2 checkpoint RSI preparation contract differs')
    saved = weights.get('stage2_contract')
    if saved != env['stage2']:
        raise ValueError('Stage2 checkpoint contract differs from the environment')


def owner_tasks(packet, count):
    tasks = packet.reshape(len(packet), -1, 5)
    ids = torch.arange(count, device=packet.device)[None, :, None]
    matches = tasks[..., 0].bool()[:, None] & (tasks[:, None, :, 2].long() == ids)
    return (tasks[:, None]*matches[..., None]).sum(-2)
