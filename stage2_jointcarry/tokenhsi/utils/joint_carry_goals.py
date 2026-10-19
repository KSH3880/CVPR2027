import torch


GOAL360 = dict(mode='uniform_angle_radius', distance=[2.2, 2.8],
               putDown='reference_terminal_xy', clearance=.1)


def clear_terminal_goals(boxes, sizes, xy, clearance):
    radius = sizes[..., :2].norm(dim=-1) / 2
    blocked = ((xy[:, None] - boxes[:, 1:, :2]).norm(dim=-1)
               < radius[:, :1] + radius[:, 1:] + clearance).any(-1)
    delta = xy[blocked] - boxes[blocked, 0, :2]
    angle = torch.atan2(delta[:, 1], delta[:, 0])[:, None] + torch.pi
    angle = angle + boxes.new_tensor([-torch.pi/3, 0., torch.pi/3])
    boxes[blocked, 1:, :2] = boxes[blocked, 0, None, :2] + 4. * torch.stack(
        (angle.cos(), angle.sin()), -1)
    return boxes


def invalid_goals(xy, boxes, sizes, bodies, ontop, clearance):
    # O2 is replaced by the sampled support in ON_TOP, but remains an obstacle in AT.
    radius = sizes[..., :2].norm(dim=-1) / 2
    target_radius = torch.where(ontop, radius[:, 2], radius[:, 0])
    distance = (xy[:, None] - boxes[..., :2]).norm(dim=-1)
    occupied = torch.ones_like(distance, dtype=torch.bool)
    occupied[:, 0] = False
    occupied[:, 2] = ~ontop
    blocked = ((distance < target_radius[:, None] + radius + clearance) & occupied).any(-1)
    blocked |= ((xy[:, None] - bodies[:, :, 0, :2]).norm(dim=-1) < 1.).any(-1)
    relative = bodies - torch.cat((xy, sizes[:, 2, 2:3] / 2), -1)[:, None, None]
    inside = (relative.abs() < sizes[:, None, None, 2] / 2 + clearance).all(-1)
    blocked |= ontop & inside.flatten(1).any(-1)
    return blocked


def sample_goals(boxes, sizes, bodies, ontop, spec):
    count = len(boxes)
    xy = boxes.new_empty(count, 2)
    pending = torch.arange(count, device=boxes.device)
    lo, hi = spec['distance']
    for _ in range(64):
        if not len(pending):
            return xy
        angle = torch.rand(len(pending), device=boxes.device) * (2 * torch.pi)
        distance = lo + torch.rand(len(pending), device=boxes.device) * (hi - lo)
        xy[pending] = boxes[pending, 0, :2] + distance[:, None] * torch.stack(
            (angle.cos(), angle.sin()), -1)
        bad = invalid_goals(xy[pending], boxes[pending], sizes[pending],
                            bodies[pending], ontop[pending], spec['clearance'])
        pending = pending[bad]
    if len(pending):
        raise RuntimeError('Joint goal360 placement failed after 64 attempts')
    return xy
