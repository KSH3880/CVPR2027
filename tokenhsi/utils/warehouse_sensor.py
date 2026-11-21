import math

import torch

from utils.before_spec import sample_roles, task_objects
from utils.geometry import rotate


def validate_config(cfg, profiles):
    for name in ('size_range', 'push_distance', 'source_depth', 'goal_clearance', 'waiting_clearance'):
        low, high = cfg[name]
        if not 0 < low <= high or not math.isfinite(low+high):
            raise ValueError('Invalid warehouse range: '+name)
    for name in ('size_step', 'floor_height', 'wall_height', 'wall_thickness', 'wall_ttc_discount',
                 'wall_box_gap', 'opening_overlap', 'open_clearance', 'pusher_stand_off',
                 'source_lateral', 'source_wall_clearance', 'goal_lateral', 'waiting_lateral'):
        if not 0 < cfg[name] or not math.isfinite(cfg[name]):
            raise ValueError('Invalid warehouse setting: '+name)
    if cfg.get('wall_cpa', 'segment') not in ('segment', 'box'):
        raise ValueError('Warehouse wall_cpa must be segment or box')
    proximity = cfg.get('wall_proximity')
    if proximity is not None:
        for key in ('distance', 'coefficient'):
            value = proximity.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid wall_proximity ' + key)
        if type(proximity.get('payload')) is not bool:
            raise ValueError('wall_proximity payload must be boolean')
        if type(proximity.get('push_box', False)) is not bool:
            raise ValueError('wall_proximity push_box must be boolean')
    for name in ('layout_candidates', 'human_candidates'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError('Warehouse candidate counts must be positive integers')
    widths = [row[1] for row in profiles]
    if (min(widths)-2*cfg['opening_overlap'] <= .7 or cfg['wall_box_gap'] >= .35
            or cfg['size_range'][0] <= max(widths)-2*cfg['opening_overlap']
            or cfg['push_distance'][1] < max(widths)-cfg['opening_overlap']+cfg['open_clearance']
            or cfg['pusher_stand_off'] < max(widths)/2+.45):
        raise ValueError('Warehouse must start blocked and allow sideways opening')

    start = cfg.get('start_distribution')
    if start:
        probabilities = start['probabilities']
        if len(probabilities) != 3 or any(not math.isfinite(v) or v < 0 for v in probabilities) or abs(sum(probabilities)-1) > 1e-6:
            raise ValueError('Warehouse closed/partial/open probabilities must sum to one')
        low, high = start['partial_open_progress']
        if not 0 < low <= high < 1:
            raise ValueError('Warehouse partial progress must stay between zero and one')
    if cfg.get('cooperative'):
        from utils.warehouse_coop import validate
        validate(cfg['cooperative'], profiles)
        done = cfg['cooperative'].get('leader_done_probability', 0.)
        if done and (not 0 < done <= 1 or not start or list(cfg['cooperative']['pusher_probabilities']) != [0., 1., 0.]
                     or cfg.get('worker_pairing_probability') != 1. or cfg['cooperative'].get('before_pairs_probability', 0.)):
            raise ValueError('leader_done_probability needs the two-pusher, one-pair warehouse with door starts and no BEFORE pairs')


def sample_roles_graph(sizes, ranges, probabilities, preset, pairing_probability):
    n, m = sizes.shape[:2]
    rows = torch.arange(n, device=sizes.device)
    roles = torch.rand(n, m, device=sizes.device).argsort(-1)
    others = roles[:, 1:]
    weights = [probabilities[key] for key in ('SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')]
    pre, tasks = sample_roles(sizes[rows[:, None], others], ranges,
        'all_solo' if preset == 'push' else preset, weights, pairing_probability)
    templates = torch.full((n, m), 5, device=sizes.device, dtype=torch.long)
    templates[rows[:, None], others] = tasks
    predecessor = torch.full_like(templates, -1)
    row, follower = (pre >= 0).nonzero(as_tuple=True)
    predecessor[row, others[row, follower]] = others[row, pre[row, follower]]
    groups = torch.arange(m, device=sizes.device)[None].expand(n, -1)
    groups = torch.where(predecessor >= 0, predecessor, groups)
    return groups, predecessor, templates, roles


def walls(half_size, thickness, height, opening):
    n = len(half_size)
    centers = half_size.new_zeros(n, 5, 3)
    half = torch.full_like(centers, height/2)
    centers[..., 2] = height/2
    centers[:, 0, 0] = half_size[:, 0]+thickness/2
    centers[:, 1:3, 1] = (half_size[:, 1:]+thickness/2)*half_size.new_tensor([-1., 1.])
    half[:, 0, 0], half[:, 0, 1] = thickness/2, half_size[:, 1]+thickness
    half[:, 1:3, 0], half[:, 1:3, 1] = half_size[:, :1], thickness/2
    outer, mouth = half_size[:, 1]+thickness, opening/2
    centers[:, 3:, 0] = -half_size[:, :1]-thickness/2
    centers[:, 3:, 1] = ((outer+mouth)/2)[:, None]*half_size.new_tensor([-1., 1.])
    half[:, 3:, 0], half[:, 3:, 1] = thickness/2, ((outer-mouth)/2)[:, None]
    starts = torch.stack((half_size, -half_size,
        torch.stack((-half_size[:, 0], half_size[:, 1]), -1), -half_size,
        torch.stack((-half_size[:, 0], mouth), -1)), 1)
    ends = torch.stack((torch.stack((half_size[:, 0], -half_size[:, 1]), -1),
        torch.stack((half_size[:, 0], -half_size[:, 1]), -1), half_size,
        torch.stack((-half_size[:, 0], -mouth), -1),
        torch.stack((-half_size[:, 0], half_size[:, 1]), -1)), 1)
    return centers, half, starts, ends


def opening_clearance(local_y, quaternion, dimensions, room_yaw, opening):
    axis = torch.stack((-room_yaw.sin(),room_yaw.cos(),torch.zeros_like(room_yaw)),-1)
    inverse = quaternion.clone()
    inverse[...,:3] *= -1
    extent = (rotate(inverse,axis).abs()*dimensions/2).sum(-1)
    return local_y.abs()-extent-opening/2


def _candidates(templates, predecessor, roles, sizes, support_sizes, room_half, blocker, near, cfg, start_mode=None):
    k, m = templates.shape
    count = cfg['layout_candidates']

    def repeat(value):
        return value[:, None].expand(-1, count, *value.shape[1:]).reshape(-1, *value.shape[1:])

    templates, predecessor, roles, sizes, support_sizes, room_half, blocker, near = map(repeat,
        (templates, predecessor, roles, sizes, support_sizes, room_half, blocker, near))
    n = len(templates)
    rows = torch.arange(n, device=templates.device)
    pusher = roles[:, 0]
    sizes = sizes.clone()
    sizes[rows, pusher] = blocker
    valid = task_objects(predecessor, templates)
    random = lambda *shape: torch.rand(*shape, device=templates.device)
    door = sizes.new_zeros(n, 2)
    door[:, 0] = -cfg['wall_thickness']-cfg['wall_box_gap']-blocker[:, 0]/2
    sign = torch.where(random(n) < .5, -1., 1.)
    low = (blocker[:, 1]-cfg['opening_overlap']+cfg['open_clearance']).clamp_min(cfg['push_distance'][0])
    distance = low+random(n)*(cfg['push_distance'][1]-low)
    landing = door.clone()
    landing[:, 1] = sign*distance
    start = door.clone()
    if start_mode is not None:
        mode = repeat(start_mode)
        low, high = cfg['start_distribution']['partial_open_progress']
        progress = low+random(n)*(high-low)
        required = blocker[:,1]-cfg['opening_overlap']+cfg['open_clearance']
        moved = torch.where(mode == 1, progress*required, torch.where(mode == 2, distance, 0.))
        start[:,1] += sign*moved
    roots = sizes.new_zeros(n, m, 2)
    roots[rows, pusher] = start
    roots[rows, pusher, 1] -= sign*cfg['pusher_stand_off']

    source = sizes.new_empty(n, m, 2)
    far = (2*room_half[:, 0]-.8).clamp_max(cfg['source_depth'][1])
    source[..., 0] = cfg['source_depth'][0]+random(n, m)*(far[:, None]-cfg['source_depth'][0])
    side = (room_half[:, 1, None]-sizes[..., 1]/2-cfg['source_wall_clearance']).clamp_max(cfg['source_lateral'])
    source[..., 1] = (2*random(n, m)-1)*side
    source[rows, pusher] = start

    def outside():
        xy = sizes.new_empty(n, m, 2)
        low, high = cfg['goal_clearance']
        xy[..., 0] = door[:, 0, None]-blocker[:, 0, None]/2-low-random(n, m)*(high-low)
        xy[..., 1] = (2*random(n, m)-1)*cfg['goal_lateral']
        return xy

    goal = torch.where((templates == 3)[..., None], outside(), source)
    goal[rows, pusher] = landing
    support = outside()
    leader = predecessor.clamp_min(0)
    goal = torch.where(((predecessor >= 0) & (templates < 3))[...,None],
        source.gather(1,leader[...,None].expand(-1,-1,2)),goal)
    shared = goal.gather(1, leader[..., None].expand(-1, -1, 2))
    target = torch.where((templates == 3)[..., None], goal,
        torch.where((predecessor >= 0)[..., None], shared, support))
    moving = (templates == 3) | (templates == 4)
    cap = torch.minimum(far[:,None],target[...,0]+5.)
    source_x = cfg['source_depth'][0]+random(n,m)*(cap-cfg['source_depth'][0]).clamp_min(0)
    reach = (25.-(source_x-target[...,0]).square()).clamp_min(0).sqrt()
    bottom = torch.maximum(-side,target[...,1]-reach)
    top = torch.minimum(side,target[...,1]+reach)
    source_y = bottom+random(n,m)*(top-bottom).clamp_min(0)
    source = torch.where(moving[...,None],torch.stack((source_x,source_y),-1),source)
    transport = (source-target).norm(dim=-1)
    good = (~moving | ((transport >= 1.) & (transport <= 5.))).all(-1)
    good &= (~moving | ((cap >= cfg['source_depth'][0]) & (top >= bottom))).all(-1)
    neutral = templates != 3
    neutral[rows,pusher] = False
    goal = torch.where(neutral[...,None],source,goal)
    goal = torch.where(((predecessor >= 0) & (templates < 3))[...,None],
        source.gather(1,leader[...,None].expand(-1,-1,2)),goal)
    target_size = torch.where((templates == 3)[..., None], sizes,
        torch.where((predecessor >= 0)[..., None], sizes.gather(1, leader[..., None].expand(-1,-1,3)), support_sizes))
    clear = (target-landing[:, None]).abs() >= (target_size[..., :2]+blocker[:, None, :2])/2+.25
    good &= (~moving | clear.any(-1)).all(-1)

    positions = torch.cat((source, support), 1)
    dimensions = torch.cat((sizes, support_sizes), 1)
    radii = dimensions[..., :2].norm(dim=-1)/2
    separation = (positions[:, :, None]-positions[:, None]).norm(dim=-1)
    pairs = valid[:, :, None] & valid[:, None] & torch.triu(torch.ones(2*m,2*m,device=templates.device,dtype=torch.bool),1)
    good &= ~((separation < radii[:, :, None]+radii[:, None]+.052) & pairs).flatten(1).any(-1)
    own = torch.arange(2*m, device=templates.device)[None] == pusher[:, None]
    nearest = ((positions-roots[rows, pusher, None]).abs()-dimensions[..., :2]/2).clamp_min(0).norm(dim=-1)
    good &= ~(valid & ~own & (nearest < .5)).any(-1)
    opened_clear = ((positions-landing[:,None]).abs() >= (dimensions[...,:2]+blocker[:,None,:2])/2+.15).any(-1)
    good &= (~valid | own | opened_clear).all(-1)
    sweep = blocker[:,:2].clone()/2
    sweep[:,1] += distance/2
    swept_clear = ((positions-(door+landing)[:,None]/2).abs() >= dimensions[...,:2]/2+sweep[:,None]+.15).any(-1)
    good &= (~valid | own | swept_clear).all(-1)
    goal_distance = (goal[:, :, None]-positions[:, None]).norm(dim=-1)
    object_owner = torch.arange(2*m,device=templates.device)[None,None] != torch.arange(m,device=templates.device)[None,:,None]
    good &= ~((templates == 3)[...,None] & valid[:,None] & object_owner &
        (goal_distance < radii[:,:m,None]+radii[:,None]+.252)).flatten(1).any(-1)

    chosen = good.nonzero().flatten()
    assigned = torch.zeros(n, m, device=templates.device, dtype=torch.bool)
    assigned[rows, pusher] = True
    workers = roles[:, 1:]
    order = (predecessor.gather(1, workers) >= 0).long().argsort(dim=-1, stable=True)
    workers = workers.gather(1, order)
    for column in range(m-1):
        if not len(chosen):
            break
        person = workers[chosen, column]
        leading = leader[chosen, person]
        follower = predecessor[chosen, person] >= 0
        task = templates[chosen, person]
        points = sizes.new_empty(len(chosen), cfg['human_candidates'], 2)
        front = door[chosen, 0]-blocker[chosen, 0]/2
        low, high = cfg['waiting_clearance']
        at = follower & (task < 3)
        point_target = torch.where(at[:,None], goal[chosen,leading], source[chosen,person])
        leading_route = (roots[chosen,leading]-source[chosen,leading]).norm(dim=-1)+(source[chosen,leading]-goal[chosen,leading]).norm(dim=-1)
        carried = torch.where(task == 4,transport[chosen,person],0.)
        limit = torch.where(at,10.,5.)
        constrained = follower & near[chosen,person]
        limit = torch.where(constrained,torch.minimum(limit,leading_route-carried-.001),limit)
        left = torch.maximum(front-high,point_target[:,0]-limit)
        right = torch.minimum(front-low,point_target[:,0]+limit)
        points[...,0] = left[:,None]+random(len(chosen),cfg['human_candidates'])*(right-left).clamp_min(0)[:,None]
        reach = (limit[:,None].square()-(points[...,0]-point_target[:,0,None]).square()).clamp_min(0).sqrt()
        bottom = (point_target[:,1,None]-reach).clamp_min(-cfg['waiting_lateral'])
        top = (point_target[:,1,None]+reach).clamp_max(cfg['waiting_lateral'])
        points[...,1] = bottom+random(len(chosen),cfg['human_candidates'])*(top-bottom).clamp_min(0)
        approach = (points-point_target[:,None]).norm(dim=-1)
        acceptable = (approach >= 1.) & (approach <= limit[:,None])
        acceptable &= (right >= left)[:,None] & (top >= bottom)
        route = approach+torch.where(task == 4,transport[chosen,person],0.)[:,None]
        acceptable &= ~(follower & near[chosen,person])[:,None] | (route < leading_route[:,None]-.001)
        acceptable &= ~follower[:,None] | ((points-goal[chosen,leading,None]).norm(dim=-1) >= 1.001)
        acceptable &= ~(((points[:,:,None]-roots[chosen,None]).norm(dim=-1) < .85) & assigned[chosen,None]).any(-1)
        reach = ((points[:,:,None]-positions[chosen,None]).abs()-dimensions[chosen,None,:,:2]/2).clamp_min(0).norm(dim=-1)
        acceptable &= ~((reach < .5) & valid[chosen,None]).any(-1)
        success = acceptable.any(-1)
        roots[chosen,person] = points[torch.arange(len(chosen),device=templates.device),acceptable.long().argmax(-1)]
        assigned[chosen,person] = success
        good[chosen] &= success
        chosen = chosen[success]

    feasible = good.view(k,count)
    index = torch.arange(k,device=templates.device)*count+feasible.long().argmax(-1)
    result = dict(root=roots[index],source=source[index],support=support[index],goal=goal[index],
        door=door[index],landing=landing[index],sign=sign[index])
    return result, feasible.any(-1)


def sample_layout(templates, predecessor, roles, sizes, support_sizes, room_half, blocker, near, cfg, tries, start_mode=None):
    inputs = (templates,predecessor,roles,sizes,support_sizes,room_half,blocker,near)
    pending = torch.arange(len(templates),device=templates.device)
    accepted = torch.zeros(len(templates),device=templates.device,dtype=torch.bool)
    result = None
    for _ in range(tries):
        trial, good = _candidates(*(value[pending] for value in inputs),cfg,
            None if start_mode is None else start_mode[pending])
        if result is None:
            result = {key: value.new_zeros(len(templates),*value.shape[1:]) for key,value in trial.items()}
        for key,value in trial.items():
            result[key][pending[good]] = value[good]
        accepted[pending[good]] = True
        pending = pending[~good]
        if not len(pending):
            break
    result['valid'] = accepted
    return result
