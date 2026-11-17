from copy import deepcopy

import torch
from utils import stage2_spec as stage2
from utils.before_spec import sample_roles

SCENES = ('cliff', 'bridge', 'room')
CLIFF, BRIDGE, ROOM = 0, 1, 2
FLOOR, SURFACE_A, SURFACE_B = 0, 1, 2
PUSH = 5
ROOM_SAMPLING = dict(before_pairing_probability=1.0, independent_push=False)


def validate_sensor(env):
    if (env['numAgents'], env['numObjects']) != (4, 8):
        raise ValueError('Sensor scenes require four people and eight tokens')
    scene = env['sensorScene']
    bridge = scene['bridge']
    if bridge.get('layout_cache') is not None:
        from utils.episode_layout_cache import validate as validate_layout_cache
        validate_layout_cache(env)
    if scene.get('variant') in ('cliff_bridge_compact_v1', 'bridge_before_fixed_v1'):
        from utils.compact_bridge import validate
        validate(scene)
    if scene.get('variant') == 'bridge_before_fixed_v1':
        from utils.bridge_before_fixed import validate as validate_fixed
        validate_fixed(scene)
    if bridge.get('carry_boundary_rsi') is not None:
        from utils.bridge_before_carry_rsi import validate as validate_carry
        validate_carry(env)
    if bridge.get('humanoid_height_reference', 'terrain') not in ('terrain', 'fixed_start_surface'):
        raise ValueError('Unsupported Bridge humanoid height reference')
    surface = bridge.get('top_surface')
    if surface is not None and surface != dict(thickness=.005, friction=1.0):
        raise ValueError('Bridge top surface requires thickness0.005 and friction1.0')
    sampling = scene['room'].get('sampling')
    if sampling is not None and sampling != ROOM_SAMPLING:
        raise ValueError('Unsupported Room sampling settings')
    if scene['before_leader_size'] != [[.4,.6],[.4,.6],[.35,.5]]:
        raise ValueError('BEFORE leaders require the shared size intersection')
    if scene['bridge']['pusher_counts'] != [1,2,3]:
        raise ValueError('Bridge pusher counts must be 1/2/3')
    spec = scene['bridge'].get('start_distribution')
    if scene['bridge'].get('curriculum'):
        from utils.bridge_curriculum import validate
        validate(scene['bridge']['curriculum'])
    if spec:
        probabilities = spec['probabilities']
        if len(probabilities) != 3 or min(probabilities) < 0 or abs(sum(probabilities)-1) > 1e-6:
            raise ValueError('Bridge present/near/insert probabilities must sum to one')
        if not .01 <= spec['entry_insert_depth'] <= .02:
            raise ValueError('Bridge insertion must be 1-2cm')
        for key in ('crosser_edge_distance', 'pickup_distance'):
            low, high = spec[key]
            if not 0 < low <= high:
                raise ValueError('Invalid bridge '+key)


def sample_bridge_starts(spec, count, device):
    return torch.multinomial(torch.tensor(spec['probabilities'], device=device), count, replacement=True)


def paired_people(predecessor):
    return stage2.before_members(predecessor)


def room_sampling_resume(saved, current):
    current = deepcopy(current)
    scene = current.get('sensorScene', {})
    sampling = scene.get('room', {}).pop('sampling', None)
    return sampling == ROOM_SAMPLING and bool(scene.get('bridge', {}).get('curriculum')) and saved == current


def room_roles(sizes, ranges, probabilities, domain, groups, pre, tasks, preset='random_scenario'):
    weights = list(probabilities.values())[1:5]
    rows = (domain == 0).nonzero().flatten()
    if len(rows):
        selected = preset if preset.startswith('before_') else 'random_scenario'
        p, t = sample_roles(sizes[rows], ranges, selected,
            solo_probabilities=weights, pairing_probability=1.)
        groups[rows] = torch.where(p >= 0, p, torch.arange(tasks.shape[1], device=tasks.device))
        pre[rows], tasks[rows] = p, t
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    solo_push = (tasks == PUSH) & (members == 1) & ~paired_people(pre)
    if solo_push.any():
        tasks[solo_push] = torch.multinomial(sizes.new_tensor(weights), int(solo_push.sum()), True)+1
    return groups, pre, tasks


def bridge_roles(sizes, ranges, probabilities, counts):
    n, m = sizes.shape[:2]
    rows = torch.arange(n, device=sizes.device)
    order = torch.rand(n, m, device=sizes.device).argsort(-1)
    # Keep crosser in column 1, remaining pushers in columns 2 and 3.
    roles = order.clone()
    groups = torch.arange(m, device=sizes.device)[None].repeat(n, 1)
    pre = torch.full_like(groups, -1)
    tasks = torch.multinomial(sizes.new_tensor(list(probabilities.values())), n*m, True).reshape(n,m)
    push = torch.zeros_like(groups, dtype=torch.bool)
    for position, column in enumerate((0,2,3)):
        active = counts > position
        push[rows[active], roles[active,column]] = True
    groups = torch.where(push, roles[:, :1], groups)
    tasks[push] = PUSH
    tasks[rows,roles[:,1]] = torch.randint(1,5,(n,),device=sizes.device)
    pair_rows = (counts == 1).nonzero().flatten()
    if len(pair_rows):
        others = roles[pair_rows,2:]
        local, picked = sample_roles(sizes[pair_rows[:,None],others], ranges, 'random_scenario',
            solo_probabilities=list(probabilities.values())[1:], pairing_probability=.5)
        global_pre = others.gather(1,local.clamp_min(0))
        pre[pair_rows[:,None],others] = torch.where(local>=0,global_pre,-1)
        groups[pair_rows[:,None],others] = torch.where(local>=0,global_pre,others)
        tasks[pair_rows[:,None],others] = picked
    return groups, pre, tasks, roles, push


def sample_surfaces(templates, predecessor, roles, scene, pushers):
    surface = torch.where(templates == PUSH, FLOOR, SURFACE_A)
    start = surface.clone()
    paired = paired_people(predecessor)
    unreachable = (scene == CLIFF)[:,None] & ~paired & (templates != PUSH)
    rows = (scene == BRIDGE).nonzero().flatten()
    if len(rows):
        cross = roles[rows,1]
        first = torch.randint(1,3,(len(rows),),device=scene.device)
        start[rows,cross], surface[rows,cross] = first, 3-first
        sites = torch.rand(len(rows),3,device=scene.device).argsort(-1)
        for index, column in enumerate((2,3)):
            person = roles[rows,column]
            site = sites[:,index]
            site = torch.where(paired[rows,person],sites[:,0],site)
            site = torch.where(templates[rows,person]==PUSH,FLOOR,site)
            start[rows,person] = surface[rows,person] = site
    return surface, start, unreachable
