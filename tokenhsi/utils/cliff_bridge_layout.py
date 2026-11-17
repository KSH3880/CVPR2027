import torch

from utils.before_goal_sampling import route_distances


def crossing_layout(start_center, start_half, dest_center, dest_half, landing, bridge, gap,
                    tasks, pre, radii, phases, progress, near, spec, wait_buffer, tries=128,
                    distance_alpha=None, boundary=None):
    n, m = tasks.shape
    device = tasks.device
    rows = torch.arange(n, device=device)
    active = phases >= 0
    follower = pre >= 0
    leaders = torch.zeros_like(tasks)
    leaders.scatter_add_(1, pre.clamp_min(0), follower.long())
    leaders = leaders > 0
    valid = torch.cat((active & ~(follower & (tasks < 3)),
        active & ((tasks == 3) | ((tasks == 4) & ~follower))), -1)
    radius = torch.cat((radii[..., 0], radii[..., 1]), -1)
    positions = torch.zeros(n, 2*m, 2, device=device)
    placed = torch.zeros_like(valid)
    direction = torch.sign(start_center[:, 0]-landing[:, 0])
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    slope_limit = (bridge[:, 1]/2-.25)/(gap/2)
    if distance_alpha is not None:
        from utils.compact_bridge import goal_band, source_depth
        reach = direction*(gap/2+2*start_half[:,0]-.8)
        first = (start_center[:,1]-start_half[:,1]+.8-landing[:,1])/reach
        second = (start_center[:,1]+start_half[:,1]-.8-landing[:,1])/reach
        slope_low = torch.maximum(-slope_limit,torch.minimum(first,second))
        slope_high = torch.minimum(slope_limit,torch.maximum(first,second))
        order = torch.argsort((~active).long()*100+leaders.long()*m+torch.arange(m, device=device), dim=-1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(m, device=device).expand(n, -1))
        rank = torch.where(active, rank, 0)
    for slot in (*range(m, 2*m), *range(m)):
        pending = valid[:, slot].clone()
        person = slot % m
        anchor_owner = torch.where(follower[:, person], pre[:, person], person).clamp_min(0)
        anchored = (slot < m) & (tasks[:, person] >= 3)
        aligned = (slot < m) | ((slot >= m) & leaders[:, person])
        clearance = radius[:, slot]+spec['exit_clearance']
        if distance_alpha is not None:
            clearance = radius[:, slot]+.8+.7*distance_alpha
        if slot >= m:
            clearance = torch.where(leaders[:, person],
                radius[:, slot]+wait_buffer+.8, clearance)
        for _ in range(tries):
            ids = pending.nonzero().flatten()
            if not len(ids):
                break
            candidate = dest_center[ids]+(2*torch.rand(len(ids), 2, device=device)-1)*(
                dest_half[ids]-radius[ids, slot, None]-.5).clamp_min(0)
            if distance_alpha is not None:
                goal_rank = rank[ids, person].float()
                required = torch.where(leaders[ids, person] & (slot >= m), wait_buffer+.8, .8)
                lo, hi = goal_band(distance_alpha[ids], goal_rank, radius[ids, slot],
                    2*dest_half[ids, 0], required)
                if slot < m:
                    lo = torch.where(anchored[ids], radius[ids, slot]+.8, lo)
                    hi = torch.where(anchored[ids], 2*dest_half[ids,0]-radius[ids,slot]-.5, hi)
                depth = lo+(hi-lo).clamp_min(0)*torch.rand_like(lo)
                candidate[:, 0] = landing[ids, 0]-direction[ids]*(gap[ids]/2+depth)
            slope = (2*torch.rand(len(ids), device=device)-1)*slope_limit[ids]
            if distance_alpha is not None:
                slope = slope_low[ids]+(slope_high[ids]-slope_low[ids])*torch.rand(len(ids),device=device)
            lane = landing[ids, 1]+(candidate[:, 0]-landing[ids, 0])*slope
            candidate[:, 1] = torch.where(aligned[ids], lane, candidate[:, 1])
            good = ((candidate-dest_center[ids]).abs() <= dest_half[ids]-radius[ids, slot, None]-.5).all(-1)
            depth = (candidate[:, 0]-landing[ids, 0])*(-direction[ids])-gap[ids]/2
            good &= depth >= clearance[ids]
            distance = (candidate[:, None]-positions[ids]).norm(dim=-1)
            good &= ~((distance < radius[ids, slot, None]+radius[ids]+.3) & placed[ids]).any(-1)
            anchor = positions[ids, m+anchor_owner[ids]]
            length = (candidate-anchor).norm(dim=-1)
            good &= ~anchored[ids] | ((length >= 1.) & (length <= 5.))
            if distance_alpha is not None:
                good &= hi >= lo
                good &= ~anchored[ids] | (length <= 2.+2*distance_alpha[ids])
            keep = ids[good]
            positions[keep, slot] = candidate[good]
            placed[keep, slot] = True
            pending[keep] = False
        failed |= pending
    payload, target = positions[:, :m], positions[:, m:].clone()
    for person in range(m):
        ids = (follower[:, person] & (tasks[:, person] < 3)).nonzero().flatten()
        payload[ids, person] = payload[ids, pre[ids, person]]
        ids = follower[:, person].nonzero().flatten()
        target[ids, person] = target[ids, pre[ids, person]]
        solo = ~follower[:, person] & (tasks[:, person] < 3)
        target[solo, person] = payload[solo, person]
    rank = (phases == 0).long().cumsum(-1)-1
    depth = .8+.4*torch.rand(n, 1, device=device)+1.5*rank+.15*torch.rand(n, m, device=device)
    if distance_alpha is not None:
        depth = source_depth(start_half, phases, distance_alpha, boundary)
    for person in range(m):
        ids = (follower[:, person] & near[:, person]).nonzero().flatten()
        leader = pre[ids, person]
        front, back = depth[ids, person].clone(), depth[ids, leader].clone()
        depth[ids, person], depth[ids, leader] = torch.minimum(front, back), torch.maximum(front, back)
    x = landing[:, 0, None]+direction[:, None]*(gap[:, None]/2+depth)
    bridge_x = landing[:, 0, None]+direction[:, None]*bridge[:, 0, None]*(.5-progress)
    x = torch.where(phases > 0, bridge_x, x)
    aim = torch.where(follower[..., None], target, payload)
    denominator = aim[..., 0]-landing[:, 0, None]
    slope = (aim[..., 1]-landing[:, 1, None])/denominator.masked_fill(denominator.abs() < 1e-6, 1.)
    human = torch.stack((x, landing[:, 1, None]+(x-landing[:, 0, None])*slope), -1)
    inside = ((human-start_center[:, None]).abs() <= start_half[:, None]-.8).all(-1)
    failed |= ((~inside & (phases == 0)) & active).any(-1)
    distance = (human[:, :, None]-human[:, None]).norm(dim=-1)
    pairs = torch.triu(torch.ones(m, m, dtype=torch.bool, device=device), diagonal=1)
    failed |= ((distance < spec['person_clearance']) & pairs & active[:, :, None] & active[:, None]).any((1, 2))
    for person in range(m):
        leader = pre[:, person].clamp_min(0)
        first, second = route_distances(human[rows, leader], payload[rows, leader], target[rows, leader],
            human[:, person], payload[:, person], tasks[:, person])
        failed |= follower[:, person] & near[:, person] & (second+.0005 >= first)
    return payload, target, human, failed
