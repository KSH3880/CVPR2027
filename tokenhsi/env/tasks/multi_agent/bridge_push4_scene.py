import torch


def draw(center, half, margin):
    room = half - torch.as_tensor(margin, device=half.device)
    return center + (torch.rand_like(center)*2-1)*room.clamp_min(0)


def stage_layout(half_a, half_b, side, cfg, near_start=None):
    n, device = len(side), side.device
    a = torch.empty(n, 2, device=device).uniform_(*cfg['translation_range'])
    need = side[:, 1]/2 + cfg['landing_margin']
    offset = (torch.rand(n, device=device)*2-1)*(half_a[:, 1]+half_b[:, 1]-2*need)
    gap = side[:, 0]+2*cfg['clearance']
    b = a + torch.stack((half_a[:, 0]+gap+half_b[:, 0], offset), -1)
    low = torch.maximum(a[:, 1]-half_a[:, 1], b[:, 1]-half_b[:, 1])+need
    high = torch.minimum(a[:, 1]+half_a[:, 1], b[:, 1]+half_b[:, 1])-need
    landing = torch.stack((a[:, 0]+half_a[:, 0]+gap/2,
                           low+torch.rand(n, device=device)*(high-low)), -1)
    ends = torch.stack((torch.maximum(a[:, 1]+half_a[:, 1], b[:, 1]+half_b[:, 1]),
                        torch.minimum(a[:, 1]-half_a[:, 1], b[:, 1]-half_b[:, 1])), -1)
    distances = (ends-landing[:, 1:]).abs()
    which = distances.argmin(-1)
    sign = torch.where(which == 0, 1., -1.)
    approach = torch.empty(n, device=device).uniform_(*cfg['approach_distance'])
    start = landing.clone()
    start[:, 1] = ends.gather(1, which[:, None])[:, 0] + sign*(side[:, 1]/2+approach)
    if near_start is not None:
        low, high = cfg['start_mixture']['near_distance']
        distance = low + torch.rand(n, device=device)*(high-low)
        start[:, 1] = torch.where(near_start, landing[:, 1]+sign*distance, start[:, 1])
    return a, b, landing, start, sign


def _task_layout(roles, tasks, surfaces, start_surfaces, sizes, half_a, half_b, side, cfg, near_start=None, aligned_start=None):
    n, device = len(tasks), tasks.device
    rows = torch.arange(n, device=device)
    a, b, landing, bridge, sign = stage_layout(half_a, half_b, side, cfg, near_start)
    centers = torch.stack((a, a, b), 1)
    halves = torch.stack((half_a, half_a, half_b), 1)
    payload = torch.zeros(n, 4, 2, device=device)
    goals, humans = torch.zeros_like(payload), torch.zeros_like(payload)
    yaw = torch.zeros(n, 4, device=device)
    pusher = roles[:, 0]
    payload[rows, pusher], goals[rows, pusher] = bridge, landing
    humans[rows, pusher] = bridge + torch.stack((torch.zeros_like(sign), sign*1.6), -1)
    yaw[rows, pusher] = torch.where(sign > 0, torch.pi, 0.)
    placed = torch.zeros(n, 4, dtype=torch.bool, device=device)
    placed[rows, pusher] = True
    radius = sizes[..., :2].norm(dim=-1)/2
    failed = torch.zeros(n, device=device, dtype=torch.bool)
    for role in range(1, 4):
        human = roles[:, role]
        surface = surfaces[rows, human]
        human_surface = start_surfaces[rows, human]
        box_radius = radius[rows, human]
        box_half = sizes[rows, human, :2]/2
        floor = surface == 0
        left = torch.rand(n, device=device) < .5
        floor_center = torch.stack((torch.where(left, a[:, 0]-half_a[:, 0]-5.5, b[:, 0]+half_b[:, 0]+5.5),
                                    a[:, 1]+(torch.rand(n, device=device)*2-1)*3), -1)
        center = torch.where(floor[:, None], floor_center, centers[rows, surface])
        half = torch.where(floor[:, None], torch.full_like(half_a, 3.), halves[rows, surface])
        hc = torch.where(floor[:, None], floor_center, centers[rows, human_surface])
        hh = torch.where(floor[:, None], torch.full_like(half_a, 3.), halves[rows, human_surface])
        pending = torch.ones(n, device=device, dtype=torch.bool)
        for _ in range(512):
            box = draw(center, half, box_half+cfg['edge_margin'])
            goal = draw(center, half, box_half+cfg['edge_margin'])
            xy = draw(hc, hh, cfg['human_margin'])
            pushing = tasks[rows, human] == 4
            sign_push = torch.where(torch.rand(n, device=device) < .5, 1., -1.)
            low = -half[:, 1]+cfg['edge_margin']+box_half[:, 1]+.8
            high = half[:, 1]-box_half[:, 1]-cfg['edge_margin']-.45
            offset = low+torch.rand(n, device=device)*(high-low)
            box[:, 1] = torch.where(pushing, center[:, 1]+sign_push*offset, box[:, 1])
            goal_offset = offset+.45+torch.rand(n, device=device)*(
                half[:, 1]-box_half[:, 1]-cfg['edge_margin']-offset-.45)
            goal[:, 1] = torch.where(pushing, center[:, 1]+sign_push*goal_offset, goal[:, 1])
            goal[:, 0] = torch.where(pushing, box[:, 0], goal[:, 0])
            direction = goal-box
            length = direction.norm(dim=-1)
            normal = direction/length.clamp_min(1e-6)[:, None]
            push_xy = box-normal*(sizes[rows, human, 1]/2+0.8)[:, None]
            xy = torch.where(pushing[:, None], push_xy, xy)
            if role == 1 and aligned_start is not None:
                fraction = (landing[:, 0]-xy[:, 0])/(box[:, 0]-landing[:, 0])
                line_y = landing[:, 1]-fraction*(box[:, 1]-landing[:, 1])
                xy[:, 1] = torch.where(aligned_start, line_y, xy[:, 1])
            valid = (length >= torch.where(pushing, .45, .7))
            valid &= ((xy-hc).abs() <= hh-cfg['edge_margin']).all(-1)
            if role == 1 and aligned_start is not None:
                valid &= ~aligned_start | ((xy-hc).abs() <= hh-cfg['human_margin']).all(-1)
                slope = (box[:, 1]-landing[:, 1])/(box[:, 0]-landing[:, 0])
                bridge_margin = side[:, 1]/2-slope.abs()*(side[:, 0]/2+cfg['clearance'])
                valid &= ~aligned_start | (bridge_margin >= .2)
            same = placed & (surfaces == surface[:, None])
            extent = sizes[..., :2]/2+box_half[:, None]+.1
            valid &= (~same | ((payload-box[:, None]).abs() > extent).any(-1)).all(-1)
            same_h = placed & (start_surfaces == human_surface[:, None])
            valid &= (~same_h | ((humans-xy[:, None]).norm(dim=-1) > 1.)).all(-1)
            same_bh = placed & (surfaces == human_surface[:, None])
            valid &= (~same_bh | ((payload-xy[:, None]).norm(dim=-1) > radius+.35)).all(-1)
            same_hb = placed & (start_surfaces == surface[:, None])
            valid &= (~same_hb | ((humans-box[:, None]).norm(dim=-1) > box_radius[:, None]+.35)).all(-1)
            own_same = surface == human_surface
            valid &= ~own_same | ((xy-box).norm(dim=-1) > box_radius+.35)
            valid &= (~same | ((payload-goal[:, None]).abs() > extent).any(-1)).all(-1)
            swept_half = box_half+(goal-box).abs()/2
            swept_center = (box+goal)/2
            clear_path = ((payload-swept_center[:, None]).abs() > sizes[..., :2]/2+swept_half[:, None]+.1).any(-1)
            valid &= ~pushing | (~same | clear_path).all(-1)
            keep = pending & valid
            payload[rows[keep], human[keep]] = box[keep]
            goals[rows[keep], human[keep]] = goal[keep]
            humans[rows[keep], human[keep]] = xy[keep]
            yaw[rows[keep], human[keep]] = torch.where(normal[:, 1] > 0, 0., torch.pi)[keep]
            pending &= ~valid
            if not pending.any():
                break
        failed |= pending
        placed[rows, human] = True
    return dict(near=a, far=b, landing=landing, bridge=bridge, sign=sign,
                boxes=payload, goals=goals, humans=humans, yaw=yaw), failed


def task_layout(roles, tasks, surfaces, start_surfaces, sizes, half_a, half_b, side, cfg, near_start=None, aligned_start=None):
    result, failed = _task_layout(roles, tasks, surfaces, start_surfaces, sizes, half_a, half_b, side, cfg, near_start, aligned_start)
    for _ in range(16):
        if not failed.any():
            return result
        rows = failed.nonzero().flatten()
        retry, pending = _task_layout(roles[rows], tasks[rows], surfaces[rows], start_surfaces[rows],
            sizes[rows], half_a[rows], half_b[rows], side[rows], cfg,
            None if near_start is None else near_start[rows],
            None if aligned_start is None else aligned_start[rows])
        for key in result:
            result[key][rows] = retry[key]
        failed[rows] = pending
    rows = failed.nonzero().flatten()
    raise RuntimeError('Bridge PUSH4 placement exhausted safe interior samples: '+str(dict(
        tasks=tasks[rows].tolist(), surface=surfaces[rows].tolist(), sizes=sizes[rows].tolist(),
        half_a=half_a[rows].tolist(), half_b=half_b[rows].tolist())))
