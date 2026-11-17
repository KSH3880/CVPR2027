import torch


def span_probabilities(spec, update):
    fraction = min(1., max(0., (update-spec['warmup_updates'])/spec['anneal_updates']))
    return [(1-fraction)*a+fraction*b for a, b in zip(spec['span_start'], spec['span_end'])]


def validate(spec):
    if not 0 <= spec['walk_probability'] <= 1:
        raise ValueError('Invalid bridge walk probability')
    for key in ('span_start', 'span_end'):
        p = spec[key]
        if len(p) != 3 or min(p) < 0 or abs(sum(p)-1) > 1e-6:
            raise ValueError('Invalid bridge span probabilities')
    if spec['warmup_updates'] < 0 or spec['anneal_updates'] <= 0:
        raise ValueError('Invalid bridge curriculum schedule')


def walk_layout(start_center, start_half, dest_center, dest_half, landing, bridge, gap,
                tasks, radii, phases, progress, spec, tries=256):
    n, device = len(tasks), tasks.device
    direction = torch.sign(start_center[:, 0]-landing[:, 0])
    source = torch.zeros(n, 3, 2, device=device)
    second, human = torch.zeros_like(source), torch.zeros_like(source)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    valid_second = tasks >= 3
    all_radii = torch.cat((radii[:, :, 0], radii[:, :, 1]), 1)
    valid_objects = torch.cat((torch.ones_like(valid_second), valid_second), 1)
    pairs = torch.triu(torch.ones(6, 6, dtype=torch.bool, device=device), diagonal=1)
    for _ in range(tries):
        objects = dest_center[:, None]+(torch.rand(n, 6, 2, device=device)*2-1)*(dest_half[:, None]-all_radii[..., None]-.5).clamp_min(0)
        slope_limit = (bridge[:, 1]/2-.25)/(gap/2)
        slope = (torch.rand(n, 3, device=device)*2-1)*slope_limit[:, None]
        objects[:, :3, 1] = landing[:, 1, None]+(objects[:, :3, 0]-landing[:, 0, None])*slope
        depth = torch.stack((.8+1.2*torch.rand(n, device=device), 2.5+1.5*torch.rand(n, device=device),
                             4.5+1.5*torch.rand(n, device=device)), 1)
        x = landing[:, 0, None]+direction[:, None]*(gap[:, None]/2+depth)
        on_bridge = phases > 0
        bridge_x = landing[:, 0, None]+direction[:, None]*bridge[:, 0, None]*(.5-progress)
        x = torch.where(on_bridge, bridge_x, x)
        xy = torch.stack((x, landing[:, 1, None]+(x-landing[:, 0, None])*slope), -1)
        in_start = ((xy-start_center[:, None]).abs() <= start_half[:, None]-.8).all(-1)
        good = (in_start | on_bridge).all(-1)
        good &= (((objects-dest_center[:, None]).abs() <= dest_half[:, None]-all_radii[..., None]-.5).all(-1) | ~valid_objects).all(-1)
        beyond = (objects[..., 0]-landing[:, 0, None])*(-direction[:, None])-gap[:, None]/2
        good &= ((beyond >= spec['exit_clearance']+all_radii) | ~valid_objects).all(-1)
        distance = (objects[:, :, None]-objects[:, None]).norm(dim=-1)
        clash = distance < all_radii[:, :, None]+all_radii[:, None]+.3
        good &= ~(clash & pairs & valid_objects[:, :, None] & valid_objects[:, None]).any((1, 2))
        hd = (xy[:, :, None]-xy[:, None]).norm(dim=-1)
        good &= ~((hd < spec['person_clearance']) & torch.triu(torch.ones(3, 3, dtype=torch.bool, device=device), diagonal=1)).any((1, 2))
        carry_distance = (objects[:, :3]-objects[:, 3:]).norm(dim=-1)
        good &= ((carry_distance >= 1.) | ~valid_second).all(-1)
        good &= ((carry_distance <= 5.) | ~valid_second).all(-1)
        keep = pending & good
        source[keep], second[keep], human[keep] = objects[keep, :3], objects[keep, 3:], xy[keep]
        pending &= ~good
        if not pending.any():
            break
    return source, second, human, pending
