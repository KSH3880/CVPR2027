from dataclasses import replace
import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.jointpush_geometry import rotate
from utils.stage1_jointpush import PUSH

SAMPLER = 'stage2_jointpush'
PRESETS = ('random_scenario', 'all_solo', 'all_coupled', 'sit', 'climb', 'holding_at', 'holding_ontop', 'push')


def checkpoint_contract(env):
    from copy import deepcopy
    from utils.before_push_spec import push_contract
    return dict(policy=deepcopy(env['stage2Policy']),push=push_contract(env),
        coupled=deepcopy(env['stage2JointPush']),graph=deepcopy(env['relationGraph']),
        box=deepcopy(env['box']),people=env['numAgents'],objects=env['numObjects'])


def check_checkpoint(checkpoint,env):
    if checkpoint.get('coupled_push_contract')!=checkpoint_contract(env):
        raise ValueError('Coupled PUSH checkpoint settings differ from the environment')


def validate_env(env):
    from copy import deepcopy
    from utils.experiments import native_config
    from utils.before_push_spec import validate_env as validate_before
    if (env['numAgents'],env['numObjects'])!=(4,8):
        raise ValueError('Coupled PUSH requires four people and eight object tokens')
    expected = deepcopy(native_config('stage2_coupled_push')['env']['stage2JointPush'])
    if 'goal_representation' in env['stage2JointPush']:
        expected['goal_representation'] = 'initial_region_offset_v1'
    if env['stage2JointPush'] != expected:
        raise ValueError('Unsupported coupled PUSH sampling/physics contract')
    from utils.stage1_platforms import validate_stage1_platform_config
    validate_stage1_platform_config(env)
    base=deepcopy(env)
    base.pop('stage2JointPush')
    original=native_config('stage2_before_push')['env']
    base['before']=deepcopy(original['before'])
    base['box']['reset']=deepcopy(original['box']['reset'])
    base['relationGraph']=deepcopy(original['relationGraph'])
    if env['relationGraph']['template_probabilities']!=base['relationGraph']['template_probabilities']:
        raise ValueError('Coupled solo probabilities must follow stage1_push')
    validate_before(base)


def expected_solo_fraction(people,probability):
    expected=[0.,1.]
    for n in range(2,people+1):
        expected.append((1-probability)*(1+expected[n-1])+probability*sum(expected[:n-1])/(n-1))
    return expected[people]/people


def compile_graph(spec, people, objects, device=None):
    from utils.edge_ontop_spec import select_graph
    groups = torch.arange(people, device=device)[None]
    return select_graph(compose_graph(torch.full_like(groups, 3), groups, False), 0)


def asset_bank(count, env, device):
    from utils.stage1_jointpush import box_profiles
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


def compose_graph(templates, groups, shuffle=True, region_goals=False):
    people = groups.shape[1]
    graph = compose_canonical_graph(torch.where(templates == 5, 3, templates), shuffle, 2*people)
    push = (templates == 5).gather(1, graph.edge_owner)
    leader = groups.gather(1, graph.edge_owner)
    goal_owner = graph.edge_owner if region_goals else leader
    return replace(graph,
        edge_relation=torch.where(push & (graph.edge_relation == 6), PUSH, graph.edge_relation),
        edge_dst=torch.where(push & (graph.edge_relation == 7), 3*people+goal_owner, graph.edge_dst))


def region_goal_offsets(physical_states, physical_sizes, members, region):
    """Fixed world offsets from the initial box pose; no RNG or distance change."""
    offset = torch.zeros_like(physical_sizes)
    width = physical_sizes[..., 0] / members
    offset[..., 0] = width * (region + .5) - physical_sizes[..., 0] / 2
    return rotate(physical_states[..., 3:7], offset)


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

