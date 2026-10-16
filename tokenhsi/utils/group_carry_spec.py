import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph
from utils.edge_stage2_spec import _empty_graph

GROUP_SAMPLER = 'four_agent_random_groups'
GROUP_SPEC = dict(mode='edge_composition', sampler=GROUP_SAMPLER,
    semantic_only=True, policy_task_roles=True, max_edges_per_agent=2,
    edge_capacity=8, pairing_probability=.8, ontop_probability=.5,
    shuffle_edge_order=True, shuffle_token_order=True,
    coupling='same_carry_task_payload_target', mask_unused_entities=True)
GROUP_PRESETS = ('random_scenario', 'joint_carry_at', 'joint_carry_ontop',
    'two_pairs_at', 'two_pairs_ontop', 'one_pair_at', 'one_pair_ontop',
    'all_solo_at', 'all_solo_ontop', 'mixed_groups')


def sample_groups(n, device='cpu', preset='random_scenario', generator=None):
    order = torch.rand(n, 4, device=device, generator=generator).argsort(-1)
    remaining = torch.ones(n, 4, dtype=torch.bool, device=device)
    groups = torch.full((n, 4), -1, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    pair_count = torch.zeros(n, dtype=torch.long, device=device)
    for step in range(4):
        active = remaining.any(-1)
        first = order.gather(1, remaining.gather(1, order).long().argmax(-1)[:, None]).flatten()
        remaining[rows, first] = False
        pair = active & remaining.any(-1) & (torch.rand(n, device=device, generator=generator) < .8)
        if preset in ('joint_carry_at', 'joint_carry_ontop', 'two_pairs_at', 'two_pairs_ontop'):
            pair = active & remaining.any(-1)
        elif preset.startswith('one_pair') or preset == 'mixed_groups':
            pair = active & remaining.any(-1) & (pair_count == 0)
        elif preset.startswith('all_solo'):
            pair[:] = False
        scores = torch.rand(n, 4, device=device, generator=generator).masked_fill(~remaining, -1.)
        partner = scores.argmax(-1)
        leader = torch.where(pair, torch.minimum(first, partner), first)
        groups[rows[active], first[active]] = leader[active]
        groups[rows[pair], partner[pair]] = leader[pair]
        remaining[rows[pair], partner[pair]] = False
        pair_count += pair
    return groups


def sample_graph(n, spec, device='cpu', preset='random_scenario', generator=None):
    if spec != GROUP_SPEC or preset not in GROUP_PRESETS:
        raise ValueError('Unsupported four-agent group sampler contract')
    groups = sample_groups(n, device, preset, generator)
    ontop = (torch.rand(n, 4, device=device, generator=generator) < .5).gather(1, groups)
    if preset.endswith('_at'):
        ontop[:] = False
    elif preset.endswith('_ontop'):
        ontop[:] = True
    elif preset == 'mixed_groups':
        counts = (groups[:, :, None] == groups[:, None, :]).sum(-1)
        ontop = counts == 1
    graph = _empty_graph(n, 4, 8, 8, device)
    graph.edge_src[:, 0::2] = torch.arange(4, device=device)
    graph.edge_dst[:, 0::2] = 4 + groups
    graph.edge_src[:, 1::2] = 4 + groups
    graph.edge_dst[:, 1::2] = torch.where(ontop, 8 + groups, 12 + groups)
    graph.edge_relation[:, 0::2] = HOLDING
    graph.edge_relation[:, 1::2] = torch.where(ontop, ON_TOP, AT)
    graph.edge_owner[:] = torch.arange(4, device=device).repeat_interleave(2)
    graph.edge_valid[:] = True
    graph.required_goal[:, 1::2] = True
    return permute_graph(graph, torch.rand(n, 8, device=device, generator=generator).argsort(-1))


def owner_tasks(packet, m):
    tasks = packet.reshape(len(packet), m, 5)
    matches = tasks[..., 0].bool()[:, None] & (tasks[:, None, :, 2].long() ==
        torch.arange(m, device=packet.device)[None, :, None])
    return (tasks[:, None] * matches[..., None]).sum(-2)


def entity_validity(packet, m, o):
    tasks = packet.reshape(len(packet), -1, 5).long()
    nodes = torch.arange(2*m+o, device=packet.device)
    used = ((tasks[..., 2:, None] == nodes) & tasks[..., :1, None].bool()).any(-2).any(-2)
    used[:, :m] = True
    return used


def physical_mapping(packet):
    tasks = owner_tasks(packet, 4).long()
    groups = tasks[..., 3] - 4
    ids = torch.arange(4, device=packet.device)[None]
    joint = (groups[:, :, None] == groups[:, None, :]).sum(-1) == 2
    leaders = groups == ids
    pair_leaders = joint & leaders
    pair_index = pair_leaders.long().cumsum(-1) - 1
    payload = torch.where(pair_leaders, 8 + pair_index, ids.expand_as(groups))
    support = torch.where(pair_leaders, 10 + pair_index, 4 + ids.expand_as(groups))
    mapping = torch.cat((payload, support), -1)
    return mapping, groups, joint
