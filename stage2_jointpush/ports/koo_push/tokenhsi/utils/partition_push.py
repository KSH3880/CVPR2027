"""Variable Push groups and fixed eight-token scene geometry, without a simulator."""
import torch
from itertools import permutations


def exclusive_regions(distances, same_group):
    """Minimum-total-distance bijection; regions from other groups are forbidden."""
    agents = distances.shape[-1]
    if distances.shape[-2] != agents or not 1 <= agents <= 4:
        raise ValueError('Exclusive assignment supports one to four agents')
    choices = torch.tensor(list(permutations(range(agents))), device=distances.device)
    rows = torch.arange(agents, device=distances.device)
    costs = distances.masked_fill(~same_group, float('inf'))[:, rows, choices].sum(-1)
    best = costs.argmin(-1)
    if not torch.isfinite(costs.gather(1, best[:, None])).all():
        raise ValueError('No valid exclusive region assignment within the groups')
    return choices[best]


def sample_surround_spawn(boxes, sizes, distance_range, min_separation=1.2):
    """Independent full-circle positions/headings outside each assigned box."""
    lo, hi = distance_range
    if not 0 <= lo <= hi or min_separation <= 0:
        raise ValueError('Invalid surround spawn distance/separation')
    n, agents = boxes.shape[:2]
    positions = boxes.new_empty(n, agents, 2)
    boundary = sizes[..., :2].norm(dim=-1) / 2
    for agent in range(agents):
        pending = torch.arange(n, device=boxes.device)
        for _ in range(128):
            angle = torch.rand(len(pending), device=boxes.device) * (2*torch.pi)
            radius = boundary[pending, agent] + lo + (hi-lo)*torch.rand(len(pending), device=boxes.device)
            candidate = boxes[pending, agent, :2] + radius[:, None]*torch.stack((angle.cos(), angle.sin()), -1)
            valid = ((candidate[:, None] - positions[pending, :agent]).norm(dim=-1)
                     >= min_separation).all(-1)
            positions[pending[valid], agent] = candidate[valid]
            pending = pending[~valid]
            if not len(pending):
                break
        else:
            raise RuntimeError('Could not sample separated surround spawns')
    heading = torch.rand(n, agents, device=boxes.device) * (2*torch.pi)
    return positions, heading


def sample_groups(n, probability, device='cpu', preset='random_scenario'):
    """Shared-support sampling: shuffle humans, then consume 1 or 2..remaining."""
    order = torch.rand(n, 4, device=device).argsort(-1)
    groups = torch.full_like(order, -1)
    cursor = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    rank = torch.arange(4, device=device)[None]
    for _ in range(4):
        remaining = 4-cursor
        coupled = (remaining >= 2) & (torch.rand(n, device=device) < probability)
        if preset == 'all_solo': coupled[:] = False
        count = torch.where(coupled,
            2+(torch.rand(n, device=device)*(remaining-1).clamp_min(0)).long(), 1)
        if preset == 'all_shared': count = remaining
        active = remaining > 0
        leader = order[rows, cursor.clamp_max(3)]
        chosen = active[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+count)[:, None])
        ordered = groups.gather(1, order)
        groups.scatter_(1, order, torch.where(chosen, leader[:, None], ordered))
        cursor += count*active
    return groups


def group_layout(groups, bank_offset=8):
    same = groups[..., None] == groups[:, None, :]
    counts = same.sum(-1)
    ids = torch.arange(4, device=groups.device)
    ranks = (same & (ids[None, None, :] < ids[None, :, None])).sum(-1)
    # Four candidate leaders x group sizes1..4, after the eight ordinary boxes.
    physical = bank_offset+4*groups+counts-1
    return physical, counts, ranks


def sample_asset_bank(n, spec, device='cpu', task_only=False):
    lo = torch.tensor(spec['size_min'], device=device)
    hi = torch.tensor(spec['size_max'], device=device)
    bank_size = 16 if task_only else 24
    sizes = lo+torch.rand(n, bank_size, 3, device=device)*(hi-lo)
    members = torch.arange(1,5,device=device).repeat(4)
    if not task_only: members = torch.cat((torch.ones(8, device=device), members))
    sizes[..., 0] *= members
    mass_lo, mass_hi = spec['mass_per_agent']
    masses = (mass_lo+torch.rand(n,bank_size,device=device)*(mass_hi-mass_lo))*members
    return sizes, masses


def group_tokens(states, sizes, assignment, keep):
    """n strips per n-person physical group + four unassigned ordinary boxes."""
    from utils.door_reference import rotate
    _, count, rank = group_layout(assignment)
    parents = torch.cat((assignment, keep), -1)
    rows = torch.arange(len(states), device=states.device)[:, None]
    physical = states[rows, parents]
    token_sizes = sizes[rows, parents].clone()
    offsets = physical.new_zeros(len(states),parents.shape[1],3)
    offsets[:, :4, 0] = token_sizes[:, :4, 0]*((rank+.5)/count-.5)
    token_sizes[:, :4, 0] /= count
    delta = rotate(physical[...,3:7], offsets)
    tokens = physical.clone()
    tokens[..., :3] += delta
    tokens[..., 7:10] += torch.cross(tokens[...,10:13], delta, dim=-1)
    return tokens, token_sizes, offsets


def nearest_boxes(roots, boxes):
    return torch.cdist(roots[..., :2], boxes[..., :2]).argmin(-1)


def strip_geometry(states, sizes, count=4):
    from utils.door_reference import rotate
    offsets = states.new_zeros(*sizes.shape[:-1], count, 3)
    offsets[..., 0] = sizes[..., 0, None] * (
        (torch.arange(count, device=states.device) + .5) / count - .5)
    q = states[..., None, 3:7].expand(*offsets.shape[:-1], 4)
    delta = rotate(q, offsets)
    tokens = states[..., None, :].expand(*offsets.shape[:-1], 13).clone()
    tokens[..., :3] += delta
    tokens[..., 7:10] += torch.cross(tokens[..., 10:13], delta, dim=-1)
    token_sizes = sizes[..., None, :].expand_as(offsets).clone()
    token_sizes[..., 0] /= count
    return tokens.flatten(1, 2), token_sizes.flatten(1, 2)


def partition_reward(roots, hands, boxes, goals, sizes, assignment, local_hands, spec):
    from utils.door_reference import rotate
    rows = torch.arange(len(roots), device=roots.device)[:, None]
    box = boxes[rows, assignment]
    size = sizes[rows, assignment]
    targets = box[..., None, :3] + rotate(
        box[..., None, 3:7].expand(*local_hands.shape[:-1], 4), local_hands)
    distance = (hands - targets).norm(dim=-1).amax(-1)
    contact_phi = torch.exp(-spec['hand_distance_scale'] * distance.square())
    same = assignment[..., None] == assignment[:, None, :]
    group_phi = contact_phi[:, None, :].expand_as(same).masked_fill(~same, 1).amin(-1)
    threshold = spec['satisfaction_threshold']
    contact = contact_phi >= threshold
    group_contact = (contact[:, None, :] | ~same).all(-1)
    delta = box[..., :3] - goals
    at_phi = torch.exp(-spec['at_distance_scale'] * delta.square().sum(-1))
    at = (at_phi >= threshold) & (delta[..., 2].abs() <= spec['at_z_tolerance']) & group_contact
    approach = targets.mean(-2).clone()
    local_normal = torch.zeros_like(box[..., :3]); local_normal[..., 1] = 1
    normal = rotate(box[..., 3:7], local_normal)
    approach -= normal * spec['approach_distance']
    progress_contact = 1 / (1 + (roots[..., :2] - approach[..., :2]).norm(dim=-1))
    progress_at = 1 / (1 + (delta[..., :2].norm(dim=-1) - .1).clamp_min(0))
    state = torch.stack((torch.where(contact | at, 1., contact_phi),
                         torch.where(at, 1., at_phi * group_phi)), -1)
    progress = torch.stack((torch.where(contact | at, 1., progress_contact),
                            torch.where(at, 1., progress_at * group_phi)), -1)
    success = torch.stack((contact | at, at), -1)
    reward = (state * spec['state_reward_weight'] + progress * spec['progress_reward_weight']
              + success.float() * spec['success_reward_weight']).sum(-1)
    return dict(reward=reward, phi=torch.stack((contact_phi, at_phi), -1),
        success=success, done=at, targets=targets, group_phi=group_phi,
        state=state, progress=progress)


def hidden_door_region_tokens(states, sizes, door_states=None):
    """Closed-door observation proxies followed by the four task regions.

    Explicit door_states provide independent static poses (v2). Without them,
    retain the old ordinary-token anchors for v1 checkpoint compatibility.
    No physical actor is added; geometry matches door_hold.urdf.
    """
    order = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], device=states.device)
    states = states[:, order].clone()
    sizes = sizes[:, order].clone()
    if door_states is not None:
        states[:, :4] = door_states
    states[:, :4, 2] = 0
    states[:, :4, 7:] = 0
    sizes[:, :4] = sizes.new_tensor([.9, .06, 1.96])
    return states, sizes


def sample_virtual_doors(boxes, sizes, roots):
    """Scatter four static, ground-level door poses around the task workspace."""
    n = len(boxes)
    lo = boxes[..., :2].amin(1) - 4.
    hi = boxes[..., :2].amax(1) + 4.
    result = boxes.new_zeros(n, 4, 13)
    for slot in range(4):
        pending = torch.ones(n, dtype=torch.bool, device=boxes.device)
        for _ in range(128):
            rows = pending.nonzero().flatten()
            if not len(rows): break
            xy = lo[rows] + torch.rand(len(rows), 2, device=boxes.device)*(hi[rows]-lo[rows])
            box_clear = ((xy[:, None]-boxes[rows, :, :2]).norm(dim=-1)
                         > sizes[rows, :, :2].norm(dim=-1)/2 + .75).all(-1)
            root_clear = (torch.cdist(xy[:, None], roots[rows, :, :2]).squeeze(1) > 1.5).all(-1)
            separated = ((xy[:, None]-result[rows, :slot, :2]).norm(dim=-1) > 2.).all(-1)
            good = box_clear & root_clear & separated
            result[rows[good], slot, :2] = xy[good]
            pending[rows[good]] = False
        if pending.any(): raise RuntimeError('Cannot scatter virtual doors with clearance')
    yaw = torch.rand(n, 4, device=boxes.device)*2*torch.pi
    result[..., 5] = (yaw/2).sin()
    result[..., 6] = (yaw/2).cos()
    return result
