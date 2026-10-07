import torch

SIZE_RSI_VARIANT = 'scenario_independent_stage1_unified_size_rsi'
TASKS = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')
TASK_PROBS = (.05, .10, .25, .30, .30)
SIZE_RANGES = {
    'HOLDING': [[.20, .60], [.20, .60], [.20, .60]],
    'SIT': [[.25, .80], [.25, .80], [.35, .50]],
    'CLIMB': [[.40, .80], [.40, .80], [.25, .50]],
    'HOLDING_AT': [[.20, .60], [.20, .60], [.20, .60]],
    'HOLDING_ON_TOP': [[.20, .60], [.20, .60], [.20, .60]],
    'SUPPORT': [[.50, .80], [.50, .80], [.25, .45]],
}
PROGRESS = {'kind': 'distance', 'sigma': 1., 'bbox_buffers': {
    'HOLDING': .20, 'SIT': .30, 'CLIMB': .30, 'AT': .10, 'ON_TOP': .10}}
SCREEN = {'seconds': .10, 'root_velocity_change': 3., 'box_speed': 5.,
          'root_displacement': .30, 'box_displacement': .30}


def size_key(size):
    return tuple(round(float(x) * 20) for x in size)


def task_probabilities(sizes, ranges=SIZE_RANGES):
    grid = (sizes * 20).round().long()
    weights = []
    for task, prior in zip(TASKS, TASK_PROBS):
        bounds = torch.tensor(ranges[task], device=sizes.device).mul(20).round().long()
        eligible = ((grid >= bounds[:, 0]) & (grid <= bounds[:, 1])).all(-1)
        count = (bounds[:, 1] - bounds[:, 0] + 1).prod()
        weights.append(eligible * (prior / count))
    weights = torch.stack(weights, -1)
    if (weights.sum(-1) == 0).any():
        raise ValueError('Source asset has no eligible task')
    return weights / weights.sum(-1, keepdim=True)


def sample_sizes(n, device, preset='random_scenario'):
    seeds = torch.multinomial(torch.tensor(TASK_PROBS, device=device), 2*n,
                              replacement=True).reshape(n, 2)
    if preset != 'random_scenario':
        name = {'holding': 'HOLDING', 'sit': 'SIT', 'climb': 'CLIMB',
                'holding_at': 'HOLDING_AT', 'holding_ontop': 'HOLDING_ON_TOP'}[preset]
        seeds.fill_(TASKS.index(name))
    sizes = torch.empty(n, 4, 3, device=device)
    for task_id, name in enumerate(TASKS):
        rows, owners = (seeds == task_id).nonzero(as_tuple=True)
        for axis, (lo, hi) in enumerate(SIZE_RANGES[name]):
            sizes[rows, owners, axis] = torch.randint(round(lo*20), round(hi*20)+1,
                                                      (len(rows),), device=device) / 20.
    for axis, (lo, hi) in enumerate(SIZE_RANGES['SUPPORT']):
        sizes[:, 2:, axis] = torch.randint(round(lo*20), round(hi*20)+1,
                                           (n, 2), device=device) / 20.
    return sizes


def sample_size_graph(sizes, spec, preset):
    from utils.edge_scenario_spec import compose_graph
    n = len(sizes)
    if preset == 'random_scenario':
        ids = torch.multinomial(task_probabilities(sizes[:, :2]).flatten(0, 1),
                                1).reshape(n, 2)
    else:
        name = {'holding': 'HOLDING', 'sit': 'SIT', 'climb': 'CLIMB',
                'holding_at': 'HOLDING_AT', 'holding_ontop': 'HOLDING_ON_TOP'}[preset]
        ids = torch.full((n, 2), TASKS.index(name), device=sizes.device)
    rows = [[TASKS[i] for i in row] for row in ids.tolist()]
    bindings = [[(a, a) if task == 'HOLDING_AT' else (a, 2+a)
                 if task == 'HOLDING_ON_TOP' else (a,)
                 for a, task in enumerate(row)] for row in rows]
    return compose_graph(rows, bindings, shuffle=spec['shuffle_edge_order']
                         and preset == 'random_scenario', device=sizes.device, num_objects=4)


def interaction_phase(skill, roots, feet, box, size):
    near = (roots[:, :2] - box[:2]).norm(dim=-1) <= size[:2].norm()/2 + .3
    top = box[2] + size[2]/2
    if skill == 'sit':
        active = near & (roots[:, 2] <= top + .35)
    elif skill == 'climb':
        active = near & (feet[..., 2].mean(-1) >= top - .10)
    else:
        return torch.ones(len(roots), dtype=torch.bool, device=roots.device)
    return active


def late_frames(valid, phase):
    candidates = (valid & phase).nonzero(as_tuple=False).flatten()
    count = max(1, (len(candidates)*3 + 9)//10)
    return candidates[-count:]


def screen_states(root, boxes, initial_root, initial_boxes, screen=SCREEN):
    finite = torch.isfinite(root).all(-1) & torch.isfinite(boxes).all(-1).all(-1)
    return (finite & ((root[:, 7:10] - initial_root[:, 7:10]).norm(dim=-1)
                      <= screen['root_velocity_change'])
            & (boxes[..., 7:10].norm(dim=-1).amax(-1) <= screen['box_speed'])
            & ((root[:, :3] - initial_root[:, :3]).norm(dim=-1)
               <= screen['root_displacement'])
            & ((boxes[..., :3] - initial_boxes[..., :3]).norm(dim=-1).amax(-1)
               <= screen['box_displacement']))


def resolve_skills(requested, weights, available, loco):
    selected = requested.clone()
    missing = ~available.gather(1, requested[:, None]).flatten()
    if missing.any():
        fallback = weights[missing] * available[missing]
        empty = fallback.sum(-1) == 0
        fallback[empty, loco] = 1.
        selected[missing] = torch.multinomial(fallback, 1).flatten()
    return selected, missing
