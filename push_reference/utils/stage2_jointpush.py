from dataclasses import replace
import torch

from push_reference.utils.edge_scenario_spec import compose_canonical_graph
from push_reference.utils.jointpush_geometry import rotate
from push_reference.utils.stage1_jointpush import PUSH

SAMPLER = 'stage2_jointpush'
PRESETS = ('random_scenario', 'all_solo', 'all_coupled', 'sit', 'climb', 'holding_at', 'holding_ontop', 'push')


def checkpoint_contract(env):
    from copy import deepcopy
    from push_reference.utils.stage1_jointpush import contract
    return dict(stage1=contract(env), stage2=deepcopy(env['stage2JointPush']),
                people=env['numAgents'], objects=env['numObjects'])


def check_checkpoint(checkpoint, env):
    if checkpoint.get('stage2_jointpush_contract') != checkpoint_contract(env):
        raise ValueError('Stage2 jointpush checkpoint settings do not match this environment')


def validate_env(env):
    from copy import deepcopy
    from pathlib import Path
    import yaml
    from push_reference.utils.stage1_jointpush import validate_env as validate_stage1
    spec = env['stage2JointPush']
    if (env['numAgents'], env['numObjects']) != (4, 8):
        raise ValueError('stage2_jointpush requires four people and eight object tokens')
    if (spec['coupled_probability'] != .8 or spec['physical_boxes'] != 24
            or spec['candidate_batch'] != 32 or spec['goal_half_angle_degrees'] not in (0., 45.)
            or spec['gate_face_tolerance'] != .18 or spec['initialization'] not in ('loco','stage1_rsi')):
        raise ValueError('Unexpected Stage2 jointpush sampling contract')
    baseline = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())['env']
    if spec['initialization'] == 'loco':
        if any(row != [1.]+[0.]*8 for row in env['templateRsi'].values()):
            raise ValueError('The loco config must initialize every task from locomotion')
    elif (env['templateRsi'] != baseline['templateRsi']
          or env['jointPush']['rsi'] != baseline['jointPush']['rsi']
          or spec.get('coupled_rsi_selection') != 'group'
          or not spec.get('solo_rsi_cache_directory')):
        raise ValueError('Stage2 RSI requires Stage1 probabilities/frame rules and group-level coupled selection')
    original = deepcopy(env)
    original.pop('stage2JointPush')
    original.pop('stage2')
    original.update(numAgents=2, numObjects=4, templateRsi=baseline['templateRsi'])
    original['relationGraph'].update(sampler='stage1_jointpush', edge_capacity=4)
    validate_stage1(original)


def compile_graph(spec, people, objects, device=None):
    from push_reference.utils.edge_ontop_spec import select_graph
    groups = torch.arange(people, device=device)[None]
    return select_graph(compose_graph(torch.full_like(groups, 3), groups, False), 0)


def asset_bank(count, env, device):
    from push_reference.utils.stage1_jointpush import box_profiles
    sizes = torch.empty(count, 24, 3, device=device)
    ranges = env['box']['build']['taskSizeRanges']
    for bank, name in enumerate(('HOLDING_AT','SIT','CLIMB','SUPPORT')):
        for axis, (low, high) in enumerate(ranges[name]):
            sizes[:,bank*4:bank*4+4,axis] = torch.randint(round(low*20),round(high*20)+1,(count,4),device=device)/20
    profiles = torch.tensor(box_profiles(), device=device)
    base = profiles[torch.randint(len(profiles),(count,4),device=device)]
    sizes[:,16:20] = base
    sizes[:,20:22] = base[:,:2]
    sizes[:,22:24] = base[:,:1]
    members = sizes.new_tensor([1,1,1,1,2,2,3,4])
    sizes[:,16:,0] *= members
    masses = sizes.prod(-1)*100.
    physical = env['jointPush']['box']
    if 'density' in physical:
        masses[:,16:] = sizes[:,16:].prod(-1)*physical['density']
    else:
        masses[:,16:] = physical['mass']*members
    return sizes, masses


def physical_assignment(templates, groups):
    people = torch.arange(4, device=groups.device)[None].expand_as(groups)
    counts = group_sizes(groups)
    pair_leader = (counts == 2) & (groups == people)
    pair_slot = (pair_leader.long().cumsum(-1)-1).gather(1, groups)
    push_slot = torch.where(counts == 1, 16+people,
        torch.where(counts == 2, 20+pair_slot, torch.where(counts == 3, 22, 23)))
    return torch.where(templates == 5, push_slot,
        torch.where(templates == 1, 4+people, torch.where(templates == 2, 8+people, people)))


def sample_groups(count, people, probability, device, preset='random_scenario'):
    order = torch.rand(count, people, device=device).argsort(-1)
    groups = torch.full_like(order, -1)
    cursor = torch.zeros(count, dtype=torch.long, device=device)
    rows = torch.arange(count, device=device)
    rank = torch.arange(people, device=device)[None]
    for _ in range(people):
        remaining = people-cursor
        coupled = (remaining >= 2) & (torch.rand(count, device=device) < probability)
        if preset == 'all_solo':
            coupled.zero_()
        size = torch.where(coupled,
            2+(torch.rand(count, device=device)*(remaining-1).clamp_min(0)).long(), 1)
        if preset == 'all_coupled':
            size = remaining
        active = remaining > 0
        leader = order[rows, cursor.clamp_max(people-1)]
        selected = active[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+size)[:, None])
        groups.scatter_(1, order, torch.where(selected, leader[:, None], groups.gather(1, order)))
        cursor += size*active
    return groups


def group_sizes(groups):
    return (groups[:, :, None] == groups[:, None, :]).sum(-1)


def sample_templates(groups, probabilities):
    solo = torch.multinomial(probabilities, groups.numel(), replacement=True).reshape_as(groups)
    return torch.where(group_sizes(groups) > 1, 5, solo)


def assign_regions(groups):
    people = groups.shape[1]
    order = torch.rand_like(groups, dtype=torch.float).argsort(-1)
    same = groups[:, :, None] == groups[:, None, :]
    earlier = order[:, None, :] < order[:, :, None]
    return (same & earlier).sum(-1)


def compose_graph(templates, groups, shuffle=True):
    people = groups.shape[1]
    graph = compose_canonical_graph(torch.where(templates == 5, 3, templates), shuffle, 2*people)
    push = (templates == 5).gather(1, graph.edge_owner)
    leader = groups.gather(1, graph.edge_owner)
    return replace(graph,
        edge_relation=torch.where(push & (graph.edge_relation == 6), PUSH, graph.edge_relation),
        edge_dst=torch.where(push & (graph.edge_relation == 7), 3*people+leader, graph.edge_dst))


def region_states(physical_states, physical_sizes, members, region):
    sizes = physical_sizes.clone()
    sizes[..., 0] /= members
    offset = torch.zeros_like(physical_sizes)
    offset[..., 0] = sizes[..., 0]*(region+.5)-physical_sizes[..., 0]/2
    delta = rotate(physical_states[..., 3:7], offset)
    states = physical_states.clone()
    states[..., :3] += delta
    states[..., 7:10] += torch.cross(states[..., 10:13], delta, dim=-1)
    return states, sizes


def inverse_rotate(q, vector):
    inverse = q.clone()
    inverse[..., :3] *= -1
    return rotate(inverse, vector)


def group_contact_gate(hands, boxes, sizes, goals, groups, tolerance=.18, success_distance=.2, z_tolerance=.001, goal_saturation=False):
    q = boxes[..., None, 3:7].expand(*hands.shape[:-1], 4)
    local = inverse_rotate(q, hands-boxes[..., None, :3])
    excess_x = (local[..., 0].abs()-sizes[..., None, 0]/2).clamp_min(0)
    excess_z = (local[..., 2].abs()-sizes[..., None, 2]/2).clamp_min(0)
    distance = (excess_x.square()+excess_z.square()+(local[..., 1]+sizes[..., None, 1]/2).square()).sqrt()
    normal = torch.zeros_like(sizes)
    normal[..., 1] = 1.
    normal = rotate(boxes[..., 3:7], normal)
    delta = goals-boxes[..., :3]
    at_position = (delta.norm(dim=-1) <= success_distance) & (delta[..., 2].abs() <= z_tolerance)
    aligned = (normal[..., :2]*delta[..., :2]).sum(-1) > 0
    on_face = (distance <= tolerance).all(-1) & (local[..., 1] <= 0).all(-1) & (aligned | at_position)
    same = groups[:, :, None] == groups[:, None, :]
    gate = (on_face[:, None, :] | ~same).all(-1)
    if goal_saturation:
        gate = gate | at_position
    return torch.where(group_sizes(groups) > 1, gate, True)
