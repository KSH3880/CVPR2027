import math

import torch

from utils.before_spec import sample_roles, task_objects


DIRECTIONS = ('left', 'right', 'in', 'out')


def validate(spec, profiles):
    for key, count in (('pusher_probabilities', 3), ('direction_probabilities', 4)):
        values = spec[key]
        if len(values) != count or any(not math.isfinite(x) or x < 0 for x in values) or abs(sum(values)-1) > 1e-6:
            raise ValueError('Invalid warehouse cooperative '+key)
    if not 0 <= spec.get('before_pairs_probability', 0.) <= 1:
        raise ValueError('Warehouse BEFORE-pairs probability must be between zero and one')
    if spec['box_size'] not in profiles:
        raise ValueError('Warehouse cooperative box must use a native PUSH profile')
    length, width, _ = spec['box_size']
    depth, breadth = spec['room_size']
    if (depth < spec['frontal_distance']+width/2+1.5 or breadth < 3*length+2.
            or spec['apron_length'] < 3*length+1.5 or not 0 < spec['frontal_opening_gap'] < .35
            or spec['lateral_distance'] < width+.35 or spec['pusher_face_gap'] < .5):
        raise ValueError('Warehouse cooperative layout needs clear pushing and pedestrian passages')


def sample_roles_graph(sizes, ranges, probabilities, counts, pairing_probability, independent_at=0):
    n, m = sizes.shape[:2]
    roles = torch.rand(n, m, device=sizes.device).argsort(-1)
    groups = torch.arange(m, device=sizes.device)[None].repeat(n, 1)
    tasks = torch.full_like(groups, 5)
    pre = torch.full_like(groups, -1)
    weights = [probabilities[key] for key in ('SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')]
    for count in range(1, m+1):
        rows = (counts == count).nonzero().flatten()
        people = roles[rows, count:m-independent_at]
        if not len(rows):
            continue
        local, chosen = sample_roles(sizes[rows[:, None], people], ranges, 'random_scenario',
            solo_probabilities=weights, pairing_probability=pairing_probability)
        predecessor = people.gather(1, local.clamp_min(0))
        pre[rows[:, None], people] = torch.where(local >= 0, predecessor, -1)
        tasks[rows[:, None], people] = chosen
        pushers = roles[rows, :count]
        groups[rows[:, None], pushers] = roles[rows, :1]
    if independent_at:
        rows = torch.arange(n, device=sizes.device)[:, None]
        tasks[rows, roles[:, -independent_at:]] = 3
    groups = torch.where(pre >= 0, pre, groups)
    return groups, pre, tasks, roles


def box_layout(counts, directions, modes, cfg):
    spec = cfg['cooperative']
    n = len(counts)
    sizes = counts.new_tensor(spec['box_size'], dtype=torch.float).expand(n, 3).clone()
    sizes[:, 0] *= counts
    lateral = directions < 2
    sign = torch.where(directions % 2 == 0, 1., -1.)
    vector = sizes.new_zeros(n, 2)
    vector[:, 0] = torch.where(lateral, 0., sign)
    vector[:, 1] = torch.where(lateral, sign, 0.)
    yaw = torch.atan2(vector[:, 1], vector[:, 0])-math.pi/2
    opening = torch.where(lateral, sizes[:, 1]-2*cfg['opening_overlap'],
        sizes[:, 0]+spec['frontal_opening_gap'])
    world_size = torch.where(lateral[:, None], sizes[:, :2], sizes[:, [1, 0]])
    door = sizes.new_zeros(n, 2)
    door[:, 0] = torch.where(lateral, -cfg['wall_thickness']-cfg['wall_box_gap']-sizes[:, 0]/2, -.15)
    distance = torch.where(lateral, spec['lateral_distance'], spec['frontal_distance'])
    required = torch.where(lateral, (sizes[:, 1]+opening)/2+cfg['open_clearance'],
        sizes[:, 1]/2+cfg['open_clearance']+torch.where(directions == 2, .15, cfg['wall_thickness']-.15))
    lo, hi = cfg['start_distribution']['partial_open_progress']
    partial = required*(lo+torch.rand(n, device=counts.device)*(hi-lo))
    moved = torch.where(modes == 1, partial, torch.where(modes == 2, distance, 0.))
    return dict(door=door, source=door+moved[:, None]*vector, landing=door+distance[:, None]*vector,
        vector=vector, yaw=yaw, sizes=sizes, world_size=world_size, opening=opening)


def _candidates(tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, straight_paths=False):
    k, m = tasks.shape
    candidates = cfg['layout_candidates']

    def repeat(value):
        return value[:, None].expand(-1, candidates, *value.shape[1:]).reshape(-1, *value.shape[1:])

    tasks, pre, roles, sizes, supports, near, counts, directions, modes = map(repeat,
        (tasks, pre, roles, sizes, supports, near, counts, directions, modes))
    n = len(tasks)
    rows = torch.arange(n, device=tasks.device)
    push = tasks == 5
    layout = box_layout(counts, directions, modes, cfg)
    spec = cfg['cooperative']
    random = lambda *shape: torch.rand(*shape, device=tasks.device)
    side_shift = cfg.get('viewer_side_shift', 0.)

    def lateral_samples(extent, *shape):
        value = (2*random(*shape)-1)*(extent-side_shift)
        return value+value.sign()*side_shift if side_shift else value

    source = sizes.new_empty(n, m, 2)
    source[..., 0] = cfg['source_depth'][0]+random(n, m)*(cfg['source_depth'][1]-cfg['source_depth'][0])
    side = spec['room_size'][1]/2-sizes[..., 1]/2-cfg['source_wall_clearance']
    source[..., 1] = lateral_samples(side, n, m)

    def outside():
        xy = sizes.new_empty(n, m, 2)
        xy[..., 0] = -.9-random(n, m)*2.6
        xy[..., 1] = lateral_samples(spec['room_size'][1]/2-.4, n, m)
        narrow = (directions >= 2) & (counts == 3)
        outer = spec['room_size'][1]/2-.4
        inner = (layout['world_size'][:,1]/2+.7).clamp_max(outer-.05)
        edge = torch.sign(source[...,1])*(inner[:,None]+(outer-inner[:,None])*random(n,m))
        xy[...,1] = torch.where(narrow[:,None],edge,xy[...,1])
        lateral = directions < 2
        rear = -layout['vector'][:,1,None]*(1.6+random(n,m)*(outer-1.6))
        xy[...,1] = torch.where(lateral[:,None],rear,xy[...,1])
        outward = directions == 3
        xy[...,1] = torch.where(outward[:,None],edge,xy[...,1])
        return xy

    support = outside()
    goal = torch.where((tasks == 3)[..., None], outside(), source)
    source = torch.where(push[..., None], layout['source'][:, None], source)
    goal = torch.where(push[..., None], layout['landing'][:, None], goal)
    leader = pre.clamp_min(0)
    goal = torch.where(((pre >= 0) & (tasks < 3))[..., None],
        source.gather(1, leader[..., None].expand(-1, -1, 2)), goal)
    moving = (tasks == 3) | (tasks == 4)
    shared = goal.gather(1, leader[..., None].expand(-1, -1, 2))
    destination = torch.where((tasks == 3)[..., None], goal, torch.where((pre >= 0)[..., None], shared, support))
    cap = (destination[...,0]+5.).clamp_max(cfg['source_depth'][1])
    source_x = cfg['source_depth'][0]+random(n,m)*(cap-cfg['source_depth'][0]).clamp_min(0)
    reach = (25.-(source_x-destination[...,0]).square()).clamp_min(0).sqrt()
    bottom = torch.maximum(-side,destination[...,1]-reach)
    top = torch.minimum(side,destination[...,1]+reach)
    source_y = bottom+random(n,m)*(top-bottom).clamp_min(0)
    if cfg.get('viewer_independent_at', 0):
        sweep_center = (layout['door']+layout['landing'])/2
        sweep_half = (layout['world_size']+(layout['landing']-layout['door']).abs())/2
        crosses = (source_x-sweep_center[:, :1]).abs() < sizes[..., 0]/2+sweep_half[:, :1]+.15
        clearance = sizes[..., 1]/2+sweep_half[:, 1:2]+.15
        left_length = (torch.minimum(top, -clearance)-bottom).clamp_min(0)
        right_start = torch.maximum(bottom, clearance)
        right_length = (top-right_start).clamp_min(0)
        offset = random(n,m)*(left_length+right_length)
        outside_y = torch.where(offset < left_length, bottom+offset, right_start+offset-left_length)
        source_y = torch.where(crosses & (left_length+right_length > 0), outside_y, source_y)
    source = torch.where(moving[...,None],torch.stack((source_x,source_y),-1),source)
    goal = torch.where(((tasks != 3) & ~push)[...,None],source,goal)
    goal = torch.where(((pre >= 0) & (tasks < 3))[...,None],
        source.gather(1,leader[...,None].expand(-1,-1,2)),goal)
    valid_objects = task_objects(pre, tasks)
    # One physical shared box; its aliases are not distinct collision objects.
    valid_objects[:, :m] &= ~push
    positions = torch.cat((source, support), 1)
    dimensions = torch.cat((sizes, supports), 1)
    radius = dimensions[..., :2].norm(dim=-1)/2
    sweep_center = (layout['door']+layout['landing'])/2
    sweep_half = (layout['world_size']+(layout['landing']-layout['door']).abs())/2
    clear = ((positions-sweep_center[:, None]).abs() >= dimensions[..., :2]/2+sweep_half[:, None]+.15).any(-1)
    good = (~valid_objects | clear).all(-1)
    sides = cfg.get('viewer_sides', {})
    if sides:
        leaders = (pre[:, :, None] == torch.arange(m, device=tasks.device)).any(1)
        if cfg.get('viewer_independent_at', 0):
            leaders |= (tasks == 3) & (pre < 0)
        good &= leaders.any(-1)
        if sides['box'] in ('left', 'right'):
            chosen = source[..., 1] > 0 if sides['box'] == 'left' else source[..., 1] < 0
            good &= (~leaders | chosen).all(-1)
    if side_shift:
        good &= (~valid_objects | (positions[..., 1].abs() >= side_shift)).all(-1)
        good &= (push | (goal[..., 1].abs() >= side_shift)).all(-1)
    good &= (~moving | ((cap >= cfg['source_depth'][0]) & (top >= bottom))).all(-1)
    block_radius = layout['sizes'][:, :2].norm(dim=-1)/2
    gap = (positions-layout['source'][:, None]).norm(dim=-1)
    good &= (~valid_objects | (gap >= radius+block_radius[:, None]+.052)).all(-1)
    separation = (positions[:, :, None]-positions[:, None]).norm(dim=-1)
    pairs = valid_objects[:, :, None] & valid_objects[:, None] & torch.triu(
        torch.ones(2*m, 2*m, device=tasks.device, dtype=torch.bool), 1)
    good &= ~((separation < radius[:, :, None]+radius[:, None]+.052) & pairs).flatten(1).any(-1)
    transport = (source-destination).norm(dim=-1)
    good &= (~moving | ((transport >= 1.) & (transport <= 5.))).all(-1)
    goal_distance = (goal[:, :, None]-positions[:, None]).norm(dim=-1)
    other = torch.arange(2*m, device=tasks.device)[None, None] != torch.arange(m, device=tasks.device)[None, :, None]
    good &= ~((tasks == 3)[..., None] & valid_objects[:, None] & other &
        (goal_distance < radius[:, :m, None]+radius[:, None]+.252)).flatten(1).any(-1)
    good &= (~(tasks == 3) | ((goal-layout['source'][:, None]).norm(dim=-1) >=
        radius[:, :m]+block_radius[:, None]+.252)).all(-1)
    goal_clear = ((goal-sweep_center[:,None]).abs() >= sizes[...,:2]/2+sweep_half[:,None]+.15).any(-1)
    good &= (~(tasks == 3) | goal_clear).all(-1)
    if straight_paths:
        from utils.warehouse_paths import clear_segments, wall_rectangles
        leaders = (pre[:, :, None] == torch.arange(m, device=tasks.device)).any(1)
        wall_centers, wall_half = wall_rectangles(cfg, layout['opening'])
        clear = clear_segments(source, goal, wall_centers, wall_half)
        good &= (~leaders | clear).all(-1)
    roots = sizes.new_zeros(n, m, 2)
    assigned = push.clone()
    region = (push.long().cumsum(-1)-1).clamp_min(0)
    along = (region+.5)*spec['box_size'][0]-layout['sizes'][:, :1]/2
    tangent = torch.stack((layout['yaw'].cos(), layout['yaw'].sin()), -1)
    pusher = layout['source'][:, None]-layout['vector'][:, None]*(spec['box_size'][1]/2+spec['pusher_face_gap'])
    roots = torch.where(push[..., None], pusher+along[..., None]*tangent[:, None], roots)
    order = roles.gather(1, (pre.gather(1, roles) >= 0).long().argsort(dim=-1, stable=True))
    for column in range(m):
        selected = (good & ~push[rows, order[:, column]]).nonzero().flatten()
        if not len(selected):
            continue
        person = order[selected, column]
        leading = leader[selected, person]
        follower = pre[selected, person] >= 0
        task = tasks[selected, person]
        points = outside()[selected, person, None].expand(-1, cfg['human_candidates'], -1).clone()
        points[..., 0] = -.9-random(len(selected), cfg['human_candidates'])*2.6
        points[..., 1] = lateral_samples(spec['room_size'][1]/2-.4, len(selected), cfg['human_candidates'])
        target = torch.where((follower & (task < 3))[:, None], goal[selected, leading], source[selected, person])
        approach = (points-target[:, None]).norm(dim=-1)
        limit = torch.where(follower & (task < 3), 10., 5.)
        acceptable = (approach >= 1.) & (approach <= limit[:, None])
        if sides.get('leader') in ('left', 'right'):
            chosen = points[..., 1] > 0 if sides['leader'] == 'left' else points[..., 1] < 0
            acceptable &= ~leaders[selected, person, None] | chosen
        if straight_paths:
            clear = clear_segments(points, source[selected, person, None],
                wall_centers[selected], wall_half[selected])
            acceptable &= ~leaders[selected, person, None] | clear
        first = (roots[selected, leading]-source[selected, leading]).norm(dim=-1)+(source[selected, leading]-goal[selected, leading]).norm(dim=-1)
        route = approach+torch.where(task == 4, transport[selected, person], 0.)[:, None]
        acceptable &= ~(follower & near[selected, person])[:, None] | (route < first[:, None]-.001)
        acceptable &= ~follower[:, None] | ((points-goal[selected, leading, None]).norm(dim=-1) >= 1.001)
        acceptable &= ~(((points[:, :, None]-roots[selected, None]).norm(dim=-1) < .85) & assigned[selected, None]).any(-1)
        object_gap = ((points[:, :, None]-positions[selected, None]).abs()-dimensions[selected, None, :, :2]/2).clamp_min(0).norm(dim=-1)
        acceptable &= ~((object_gap < .5) & valid_objects[selected, None]).any(-1)
        box_gap = ((points-layout['source'][selected, None]).abs()-layout['world_size'][selected, None]/2).clamp_min(0).norm(dim=-1)
        acceptable &= box_gap >= .5
        success = acceptable.any(-1)
        roots[selected, person] = points[torch.arange(len(selected), device=tasks.device), acceptable.long().argmax(-1)]
        assigned[selected, person] = success
        good[selected] &= success
    good &= assigned.all(-1)
    feasible = good.view(k, candidates)
    index = torch.arange(k, device=tasks.device)*candidates+feasible.long().argmax(-1)
    result = dict(root=roots, source=source, support=support, goal=goal, landing=layout['landing'],
        box_yaw=layout['yaw'], push_heading=torch.atan2(layout['vector'][:, 1], layout['vector'][:, 0]))
    return {key: value[index] for key, value in result.items()}, feasible.any(-1)


def sample_layout(tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, tries, straight_paths=False):
    inputs = (tasks, pre, roles, sizes, supports, near, counts, directions, modes)
    pending = torch.arange(len(tasks), device=tasks.device)
    accepted = torch.zeros(len(tasks), device=tasks.device, dtype=torch.bool)
    result = None
    for _ in range(tries):
        trial, good = _candidates(*(value[pending] for value in inputs), cfg, straight_paths)
        if result is None:
            result = {key: value.new_zeros(len(tasks), *value.shape[1:]) for key, value in trial.items()}
        for key, value in trial.items():
            result[key][pending[good]] = value[good]
        accepted[pending[good]] = True
        pending = pending[~good]
        if not len(pending):
            break
    result['valid'] = accepted
    return result
