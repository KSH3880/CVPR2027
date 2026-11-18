import math

import torch
from utils import stage2_spec as stage2
from utils.before_spec import sample_roles

SCENES = ('cliff', 'bridge', 'room')
CLIFF, BRIDGE, ROOM, WAREHOUSE = 0, 1, 2, 3
FLOOR, SURFACE_A, SURFACE_B = 0, 1, 2
PUSH = 5


def validate_sensor(env):
    if (env['numAgents'], env['numObjects']) != (4, 8):
        raise ValueError('Sensor scenes require four people and eight tokens')
    scene = env['sensorScene']
    if 'wall_collision_coefficient' in scene:
        coefficient = scene['wall_collision_coefficient']
        if not math.isfinite(coefficient) or coefficient < 0 or env['agentCollisionCoeff'] <= 0:
            raise ValueError('Wall CPA requires a nonnegative wall coefficient and positive agent coefficient')
    for key in ('wall_contact_termination', 'push_box_contact_termination'):
        duration = scene.get(key, 1.)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(key+' must be a positive duration in seconds')
    if 'push_box_collision_coefficient' in scene:
        coefficient = scene['push_box_collision_coefficient']
        if not math.isfinite(coefficient) or coefficient < 0 or env['agentCollisionCoeff'] <= 0:
            raise ValueError('PUSH-box CPA requires a nonnegative coefficient and positive agent coefficient')
    if any(key.startswith('push_box_') for key in scene) and 'warehouse' not in scene:
        raise ValueError('PUSH-box penalties require the warehouse scene')
    names = SCENES + (('warehouse',) if 'warehouse' in scene else ())
    if tuple(scene['mixture']) != names:
        raise ValueError('Sensor scene order must be cliff, bridge, room, optional warehouse')
    weights = list(scene['mixture'].values())
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.):
        raise ValueError('Sensor mixture must sum to one')
    if 'warehouse' in scene:
        from utils.warehouse_sensor import validate_config
        validate_config(scene['warehouse'], env['jointPush']['box']['profiles'])
        cache = scene['warehouse'].get('layout_cache')
        if cache and (env['stateInit'] != 'Default' or not scene['warehouse'].get('cooperative')
                      or any(type(cache[k]) is not int or cache[k] < 1 for k in ('capacity', 'samples'))
                      or type(cache['warmup_resets']) is not int or cache['warmup_resets'] < 0
                      or not 0 <= cache['refresh_probability'] <= 1):
            raise ValueError('Warehouse layout_cache needs cooperative PUSH, stateInit Default and valid sizes')
    matching = env.get('warehousePushRegionMatching', False)
    if type(matching) is not bool or (matching and not scene.get('warehouse', {}).get('cooperative')):
        raise ValueError('warehousePushRegionMatching must be a boolean and needs the cooperative Warehouse')
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
    start[scene == WAREHOUSE] = FLOOR
    return surface, start, unreachable
