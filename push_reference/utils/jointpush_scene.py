import math
import torch


def radial_xy(center, bounds):
    draw = torch.rand(*center.shape[:-1], 2, device=center.device)
    angle = draw[..., 0]*2*torch.pi
    distance = bounds[0]+draw[..., 1]*(bounds[1]-bounds[0])
    return center+distance[..., None]*torch.stack((angle.cos(), angle.sin()), -1)


def segment_distance(a, b, c, d):
    def point_distance(p, x, y):
        v = y-x
        t = ((p-x)*v).sum(-1)/v.square().sum(-1).clamp_min(1e-12)
        return (p-x-t.clamp(0, 1)[..., None]*v).norm(dim=-1)
    def cross(x, y):
        return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    ab, cd = b-a, d-c
    distance = torch.stack((point_distance(a, c, d), point_distance(b, c, d),
                            point_distance(c, a, b), point_distance(d, a, b)), -1).amin(-1)
    intersect = ((cross(ab, c-a)*cross(ab, d-a) <= 0)
                 & (cross(cd, a-c)*cross(cd, b-c) <= 0) & (cross(ab, cd).abs() > 1e-8))
    return torch.where(intersect, torch.zeros_like(distance), distance)


def corridors(boxes, goals, rear_clearance):
    xy = boxes[:, :2, :2]
    direction = goals[..., :2]-xy
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rear = xy-rear_clearance*direction
    end_rear = goals[..., :2]-rear_clearance*direction
    return torch.stack((rear, xy, rear, goals[..., :2]), 2), torch.stack((xy, goals[..., :2], end_rear, end_rear), 2)


def paths_clear(boxes, sizes, goals, roots, push, spec):
    starts, ends = corridors(boxes, goals, spec['rear_clearance'])
    radii = sizes[..., :2].norm(dim=-1)/2
    gap = segment_distance(starts[:, 0, :, None], ends[:, 0, :, None],
                           starts[:, 1, None], ends[:, 1, None]).amin((1, 2))
    valid = (gap >= spec['path_separation']) | ~push.all(-1)
    other_roots = roots.flip(1)[..., :2].unsqueeze(2)
    human_gap = segment_distance(starts, ends, other_roots, other_roots).amin(-1)
    valid &= ((human_gap >= radii[:, :2]+spec['agent_clearance']) | ~push).all(-1)
    other_boxes = boxes[:, :2, :2].flip(1).unsqueeze(2)
    box_gap = segment_distance(starts, ends, other_boxes, other_boxes).amin(-1)
    required = radii[:, :2]+radii[:, :2].flip(1)+spec['obstacle_clearance']
    valid &= ((box_gap >= required) | ~push | push.flip(1)).all(-1)
    return valid


def goal_angles_valid(boxes, goals, push, spec):
    if spec['goal_angle_degrees'] == 360.:
        return torch.ones(len(boxes), device=boxes.device, dtype=torch.bool)
    q = boxes[:, :2, 3:7]
    forward = torch.stack((-2*q[..., 2]*q[..., 3], 1-2*q[..., 2].square()), -1)
    delta = goals[..., :2]-boxes[:, :2, :2]
    cosine = (forward*delta).sum(-1)/delta.norm(dim=-1).clamp_min(1e-8)
    limit = math.cos(math.radians(spec['goal_angle_degrees']/2))
    return ((cosine >= limit-1e-6) | ~push).all(-1)


def sample_layout(roots, boxes, sizes, push, spec):
    result = boxes.clone()
    goals = boxes[:, :2, :3].clone()
    pending = push.any(-1).nonzero().flatten()
    attempts = 2048 if spec['goal_angle_degrees'] == 90. else 512
    for _ in range(attempts):
        if not len(pending):
            return result, goals
        active = push[pending]
        xy = radial_xy(roots[pending, :, :2], spec['source_distance'])
        target = radial_xy(xy, spec['goal_distance'])
        candidate = boxes[pending].clone()
        candidate[:, :2, :2] = torch.where(active[..., None], xy, candidate[:, :2, :2])
        candidate[:, :2, 2] = torch.where(active, sizes[pending, :2, 2]/2, candidate[:, :2, 2])
        yaw = torch.rand(len(pending), 2, device=boxes.device)*2*torch.pi
        q = torch.zeros(len(pending), 2, 4, device=boxes.device)
        q[..., 2], q[..., 3] = (yaw/2).sin(), (yaw/2).cos()
        candidate[:, :2, 3:7] = torch.where(active[..., None], q, candidate[:, :2, 3:7])
        candidate[:, :2, 7:] = torch.where(active[..., None], 0., candidate[:, :2, 7:])
        target3 = candidate[:, :2, :3].clone()
        target3[..., :2] = target
        valid = paths_clear(candidate, sizes[pending], target3, roots[pending], active, spec)
        valid &= goal_angles_valid(candidate, target3, active, spec)
        valid &= (((target-roots[pending, :, :2]).norm(dim=-1) >= spec['goal_min_human_distance']) | ~active).all(-1)
        accepted = pending[valid]
        result[accepted] = candidate[valid]
        goals[accepted] = target3[valid]
        pending = pending[~valid]
    raise RuntimeError('stage1_jointpush could not sample separated PUSH paths')


def support_clear(boxes, sizes, goals, push, spec):
    start, end = corridors(boxes, goals, spec['rear_clearance'])
    supports = boxes[:, 2:, :2]
    gap = segment_distance(start[:, :, :, None], end[:, :, :, None],
                           supports[:, None, None], supports[:, None, None]).amin(2)
    radius = sizes[..., :2].norm(dim=-1)/2
    required = radius[:, :2, None]+radius[:, None, 2:]+spec['obstacle_clearance']
    return ((gap >= required) | ~push[..., None]).all(1)


def place_distractors(task, ids, templates):
    push = templates == 5
    spec = task.cfg['env']['jointPush']['scene']
    radii = task._box_size[ids, ..., :2].norm(dim=-1)/2
    for _ in range(128):
        boxes = task._box_states[ids]
        valid = support_clear(boxes, task._box_size[ids], task._tar_pos[ids], push, spec)
        gap = (boxes[:, 2:, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
        gap[:, 0, 2] = float('inf')
        gap[:, 1, 3] = float('inf')
        valid &= (gap >= radii[:, 2:, None]+radii[:, None, :]+.05).all(-1)
        human_gap = (boxes[:, 2:, None, :2]-task._humanoid_root_states[ids, None, :, :2]).norm(dim=-1)
        valid &= (human_gap >= (radii[:, 2:]+.35).clamp_min(1.)[..., None]).all(-1)
        rows, owners = ((~valid) & (templates != 4) & push.any(-1)[:, None]).nonzero(as_tuple=True)
        if not len(rows):
            break
        task._box_states[ids[rows], owners+2, :2] = task._sample_arena_xy(task._env_origins[ids[rows], :2])
