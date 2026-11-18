"""Fixed cliff, bridge and room geometry, plus rigid placement of Stage2 task groups.

Structure frame: the surface (or bridge stage A) is centred at the origin and stage B lies across a +x gap.
"""
import math

import torch

from env.tasks.multi_agent.collision_reward import compute_agent_cpa_collision_penalty
from env.tasks.multi_agent.square_sensor import rotation_matrix

HUMAN_RADIUS = .35
HIDDEN = -1000.
GROUP_PAD = .3


def uniform(bounds, n, device):
    return bounds[0]+(bounds[1]-bounds[0])*torch.rand(n, device=device)


def rotate_xy(xy, yaw):
    c, s = yaw.cos()[..., None], yaw.sin()[..., None]
    return torch.cat((c*xy[..., :1]-s*xy[..., 1:], s*xy[..., :1]+c*xy[..., 1:]), -1)


def yaw_quat(yaw):
    q = torch.zeros(*yaw.shape, 4, device=yaw.device)
    q[..., 2], q[..., 3] = (yaw/2).sin(), (yaw/2).cos()
    return q


def to_world(xy, yaw, shift, origin):
    shape = (len(yaw),)+(1,)*(xy.ndim-2)
    return rotate_xy(xy, yaw.view(shape))+shift.view(*shape, 2)+origin[:, :2].view(*shape, 2)


def place_structure(bounds, clearance, jitter_range):
    """Rotate by a multiple of 90 degrees and leave the floor-task centre clear on one side."""
    n, device = len(bounds), bounds.device
    yaw = torch.randint(4, (n,), device=device)*(math.pi/2)
    corners = torch.stack((bounds[:, [0, 2]], bounds[:, [0, 3]], bounds[:, [1, 2]], bounds[:, [1, 3]]), 1)
    corners = rotate_xy(corners, yaw[:, None])
    low, high = corners.amin(1), corners.amax(1)
    axis = torch.randint(2, (n,), device=device)
    direction = torch.where(torch.rand(n, device=device) < .5, 1., -1.)
    rows = torch.arange(n, device=device)
    shift = -(low+high)/2
    shift[rows, axis] = torch.where(direction > 0, clearance-low[rows, axis], -clearance-high[rows, axis])
    shift[rows, 1-axis] += uniform(jitter_range, n, device)
    placed = corners+shift[:, None]
    # Distance from the floor-task centre to the structure's near face, along the side it occupies.
    near = torch.where(direction > 0, placed[rows, :, axis].amin(-1), -placed[rows, :, axis].amax(-1))
    return dict(yaw=yaw, shift=shift, axis=axis, direction=direction, near=near, reach=placed.norm(dim=-1).amax(-1))


def room_walls(half_size, thickness, height, passage_width):
    """Four outer walls and the central partition: box centres, half sizes and CPA wall segments."""
    n = len(half_size)
    centers = half_size.new_zeros(n, 4, 3)
    centers[:, :2, 0] = (half_size[:, :1]+thickness/2)*half_size.new_tensor([1., -1.])
    centers[:, 2:, 1] = (half_size[:, 1:]+thickness/2)*half_size.new_tensor([1., -1.])
    centers[..., 2] = height/2
    half = torch.full_like(centers, height/2)
    half[:, :2, 0], half[:, :2, 1] = thickness/2, half_size[:, 1:]
    half[:, 2:, 0], half[:, 2:, 1] = half_size[:, :1]+thickness, thickness/2
    starts = torch.stack((torch.stack((half_size[:, 0], -half_size[:, 1]), -1),
                          -half_size, torch.stack((-half_size[:, 0], half_size[:, 1]), -1),
                          -half_size), 1)
    ends = torch.stack((half_size, torch.stack((-half_size[:, 0], half_size[:, 1]), -1),
                        half_size, torch.stack((half_size[:, 0], -half_size[:, 1]), -1)), 1)
    partition_half = torch.stack((torch.full_like(half_size[:, 0], thickness/2),
                                  half_size[:, 1]-passage_width,
                                  torch.full_like(half_size[:, 0], height/2)), -1)
    center = torch.zeros_like(partition_half)
    center[:, 2] = height/2
    corners = partition_half[:, None, :2]*half_size.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    return (torch.cat((centers, center[:, None]), 1), torch.cat((half, partition_half[:, None]), 1),
            torch.cat((starts, corners), 1), torch.cat((ends, corners.roll(-1, 1)), 1))


def compute_wall_cpa_collision_penalty(root_pos, root_vel, starts, ends, min_dist, discount, dt):
    """Each wall is a stationary agent at its point nearest to the root; the agent CPA does the rest."""
    segment = ends-starts
    along = (((root_pos[:, :, None, :2]-starts[:, None])*segment[:, None]).sum(-1)
             / segment.square().sum(-1)[:, None]).clamp(0., 1.)
    nearest = starts[:, None]+along.unsqueeze(-1)*segment[:, None]
    n, agents, walls = nearest.shape[:3]
    pos = torch.cat((root_pos[:, :, None, :2], nearest), 2).view(n*agents, walls+1, 2)
    vel = torch.cat((root_vel[:, :, None, :2], torch.zeros_like(nearest)), 2).view(n*agents, walls+1, 2)
    return compute_agent_cpa_collision_penalty(pos, vel, min_dist, discount, dt)[:, 0].view(n, agents)


def compute_box_cpa_collision_penalty(root_pos, root_vel, center, quat, half, box_vel, min_dist, discount, dt):
    """The box is a moving agent at its point nearest each root; the agent CPA does the rest."""
    rot = rotation_matrix(quat)
    local = ((root_pos-center[:, None]).unsqueeze(-2)@rot[:, None]).squeeze(-2)
    nearest = center[:, None]+(rot[:, None]@torch.clamp(local, -half[:, None], half[:, None])[..., None]).squeeze(-1)
    n, agents = root_pos.shape[:2]
    pos = torch.stack((root_pos[..., :2], nearest[..., :2]), 2).view(n*agents, 2, 2)
    vel = torch.stack((root_vel[..., :2], box_vel[:, None, :2].expand(-1, agents, -1)), 2).view(n*agents, 2, 2)
    return compute_agent_cpa_collision_penalty(pos, vel, min_dist, discount, dt)[:, 0].view(n, agents)


def body_box_contact(body_pos, body_rot, contact, shapes, centers, quats, halves, tolerance):
    """[n, m] people with a force-carrying body whose collision shape lies within tolerance of a box.

    The net contact force does not name the partner, so the shape-to-box gap separates contact with
    the given boxes from self, human and object contact."""
    points, radii, owner = shapes
    rot = rotation_matrix(body_rot[..., owner, :]).view(*body_rot.shape[:2], len(owner), 3, 3)
    world = body_pos[..., owner, :]+(rot@points[..., None]).squeeze(-1)
    wall = rotation_matrix(quats).view(*quats.shape[:2], 3, 3)
    offset = world[:, :, :, None]-centers[:, None, None]
    local = (offset.unsqueeze(-2)@wall[:, None, None]).squeeze(-2)
    gap = (local.abs()-halves[:, None, None]).clamp_min(0.).norm(dim=-1).min(-1)[0]-radii
    return ((gap < tolerance) & (contact[..., owner, :].abs().sum(-1) > .1)).any(-1)


def bridge_stages(half_a, half_b, bridge, cfg, align_entry=False):
    """Per-environment stage B position, gap and worst-case bounds including the channel and pusher."""
    n, device = len(half_a), half_a.device
    gap = bridge[:, 0]+2*cfg['clearance']
    need = bridge[:, 1]/2+cfg['landing_margin']
    offset = (torch.rand(n, device=device)*2-1)*(half_a[:, 1]+half_b[:, 1]-2*need)
    if align_entry:
        offset = half_a[:, 1]-half_b[:, 1]
    far = torch.stack((half_a[:, 0]+gap+half_b[:, 0], offset), -1)
    extra = bridge[:, 1]/2+cfg['approach_distance'][1]+cfg['pusher_stand_off']+HUMAN_RADIUS
    y_low = torch.minimum(-half_a[:, 1], offset-half_b[:, 1])
    y_high = torch.maximum(half_a[:, 1], offset+half_b[:, 1])
    bounds = torch.stack((-half_a[:, 0], far[:, 0]+half_b[:, 0], y_low-extra, y_high+extra), -1)
    return far, gap, bounds


def bridge_layout(half_a, half_b, far, gap, bridge, cfg, near, side, insert_depth=0., training_mode=None):
    """side=True enters from the +y channel end, so the bridge then moves toward -y."""
    n, device = len(half_a), half_a.device
    need = bridge[:, 1]/2+cfg['landing_margin']
    offset = far[:, 1]
    low = torch.maximum(-half_a[:, 1], offset-half_b[:, 1])+need
    high = torch.minimum(half_a[:, 1], offset+half_b[:, 1])-need
    landing = torch.stack((half_a[:, 0]+gap/2, low+torch.rand(n, device=device)*(high-low)), -1)
    sign = torch.where(side, 1., -1.)
    end = torch.where(side, torch.maximum(half_a[:, 1], offset+half_b[:, 1]),
                      torch.minimum(-half_a[:, 1], offset-half_b[:, 1]))
    start = landing.clone()
    start[:, 1] = end+sign*(bridge[:, 1]/2+uniform(cfg['approach_distance'], n, device))
    near_y = landing[:, 1]+sign*uniform(cfg['start_mixture']['near_distance'], n, device)
    start[:, 1] = torch.where(near, near_y, start[:, 1])
    if insert_depth:
        start[:, 1] = end+sign*(bridge[:, 1]/2-insert_depth)
    if training_mode is not None:
        depth = cfg['start_distribution']['entry_insert_depth']
        start[:, 1] = torch.where(training_mode == 0, landing[:, 1],
            torch.where(training_mode == 1, near_y, end+sign*(bridge[:, 1]/2-depth)))
    pusher = start+torch.stack((torch.zeros_like(sign), sign*cfg['pusher_stand_off']), -1)
    return dict(landing=landing, start=start, pusher=pusher, sign=sign)


def _draw(center, half, margin):
    return center+(torch.rand_like(center)*2-1)*(half-margin[..., None]).clamp_min(0)


def crosser_layout(task, dest_center, dest_half, start_center, start_half, payload_radius, second_radius,
                   landing, bridge, gap, aligned, edge_margin, human_margin, cfg, carry_from_start=None,
                   present=None):
    n, device = len(task), task.device
    if carry_from_start is None:
        carry_from_start = torch.zeros(n, dtype=torch.bool, device=device)
    box = torch.zeros(n, 2, device=device)
    second, human = torch.zeros_like(box), torch.zeros_like(box)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    has_second = (task == 3) | (task == 4)
    margin = torch.full((n,), human_margin, device=device)
    box_center = torch.where(carry_from_start[:, None], start_center, dest_center)
    box_half = torch.where(carry_from_start[:, None], start_half, dest_half)
    for _ in range(256):
        box_try = _draw(box_center, box_half, payload_radius+edge_margin)
        second_try = _draw(dest_center, dest_half, second_radius+edge_margin)
        xy = _draw(start_center, start_half, margin)
        if present is not None:
            spec = cfg['start_distribution']
            direction = torch.sign(start_center[:, 0]-landing[:, 0])
            depth = uniform(spec['crosser_edge_distance'], n, device)
            entry_x = landing[:, 0]+direction*(gap/2+depth)
            xy[:, 0] = torch.where(present, entry_x, xy[:, 0])
        across = torch.where(carry_from_start[:, None], second_try, box_try)
        slope = (across[:, 1]-landing[:, 1])/(across[:, 0]-landing[:, 0])
        xy[:, 1] = torch.where(aligned, landing[:, 1]+(xy[:, 0]-landing[:, 0])*slope, xy[:, 1])
        box_try[:, 1] = torch.where(aligned & carry_from_start,
            landing[:, 1]+(box_try[:, 0]-landing[:, 0])*slope, box_try[:, 1])
        present_carry = torch.zeros_like(pending) if present is None else present & carry_from_start
        if present is not None:
            away = xy-across
            away = away/away.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            pickup = xy+away*uniform(spec['pickup_distance'], n, device)[:, None]
            box_try = torch.where(present_carry[:, None], pickup, box_try)
        side_margin = bridge[:, 1]/2-slope.abs()*(bridge[:, 0]/2+cfg['clearance'])
        valid = ~aligned | (((xy-start_center).abs() <= start_half-human_margin).all(-1)
                            & (side_margin >= cfg['aligned_bridge_margin']))
        valid &= ((box_try-box_center).abs() <= box_half-(payload_radius+edge_margin)[:, None]).all(-1)
        valid &= ~carry_from_start | ((xy-box_try).norm(dim=-1) >= HUMAN_RADIUS+payload_radius+GROUP_PAD)
        valid &= ~(aligned & carry_from_start & ~present_carry) | ((xy[:, 0]-box_try[:, 0])*(box_try[:, 0]-landing[:, 0]) > 0)
        keepout = gap/2+cfg['landing_box_clearance']
        valid &= (box_try-landing).norm(dim=-1) >= keepout+payload_radius
        valid &= ~has_second | ((second_try-landing).norm(dim=-1) >= keepout+second_radius)
        valid &= ~has_second | ((second_try-box_try).norm(dim=-1) >= (payload_radius+second_radius+.1).clamp_min(1.))
        keep = pending & valid
        box[keep], second[keep], human[keep] = box_try[keep], second_try[keep], xy[keep]
        pending &= ~valid
        if not pending.any():
            break
    return box, second, human, pending


def offstage_layout(half, radius, second_distance, gap, margin, max_distance):
    """Cliff object on the floor beyond a random edge, the person on stage and a floor goal further out."""
    n, device = len(half), half.device
    rows = torch.arange(n, device=device)
    side = torch.randint(4, (n,), device=device)
    axis = side % 2
    sign = torch.where(side < 2, 1., -1.)
    normal_half, tangent_half = half[rows, axis], half[rows, 1-axis]
    drawn_gap = uniform(gap, n, device)
    out = normal_half+radius+drawn_gap
    tangent = (torch.rand(n, device=device)*2-1)*(tangent_half-margin)
    minimum = radius+drawn_gap+margin
    distance = minimum+torch.rand(n, device=device)*(max_distance-minimum).clamp_min(0)
    angle = (torch.rand(n, device=device)*2-1)*math.pi/4
    depth, lateral = distance*angle.cos(), distance*angle.sin()
    straight = (depth < minimum) | ((tangent+lateral).abs() > tangent_half-margin)
    depth, lateral = torch.where(straight, distance, depth), lateral.masked_fill(straight, 0.)
    outward = (torch.rand(n, device=device)*2-1)*math.pi/2
    second_normal = out+second_distance*outward.cos()
    second_tangent = tangent+second_distance*outward.sin()

    def local(normal, along):
        xy = torch.zeros(n, 2, device=device)
        xy[rows, axis] = sign*normal
        xy[rows, 1-axis] = along
        return xy

    return local(out-depth, tangent+lateral), local(out, tangent), local(second_normal, second_tangent)


def terrain_fall(contact, body_pos, contact_ids, ground, thresholds, progress, floor_height, floor_start):
    """Surface starters fail on any floor contact; floor starters keep the ordinary fall rule."""
    contact_mask = contact.abs().sum(-1) > .1
    low = body_pos[..., 2]-ground < thresholds
    outside = ground <= floor_height[:, None, None]+1e-4
    floor_contact = (contact_mask & low & outside).any(-1) & ~floor_start
    contact_mask[..., contact_ids] = False
    low[..., contact_ids] = False
    fallen = contact_mask.any(-1) & low.any(-1)
    return (fallen | floor_contact) & (progress[:, None] > 1)


class Occupancy:
    """Discs already placed in each environment, per surface."""

    def __init__(self, n, capacity, device):
        self.xy = torch.zeros(n, capacity, 2, device=device)
        self.radius = torch.zeros(n, capacity, device=device)
        self.surface = torch.zeros(n, capacity, dtype=torch.long, device=device)
        self.valid = torch.zeros(n, capacity, dtype=torch.bool, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)

    def conflicts(self, rows, xy, radius, valid, surface):
        """xy, radius, valid: (k, entities, ...). The pad also keeps other groups' limbs off a box at reset."""
        reach = radius[:, :, None]+self.radius[rows, None]+GROUP_PAD
        close = (xy[:, :, None]-self.xy[rows, None]).norm(dim=-1) < reach
        same = self.surface[rows, None] == surface[:, None, None]
        return (close & same & self.valid[rows, None] & valid[:, :, None]).flatten(1).any(-1)

    def add(self, rows, xy, radius, valid, surface):
        for entity in range(xy.shape[1]):
            chosen = valid[:, entity].nonzero().flatten()
            if not len(chosen):
                continue
            row, slot = rows[chosen], self.count[rows[chosen]]
            self.xy[row, slot] = xy[chosen, entity]
            self.radius[row, slot] = radius[chosen, entity]
            self.surface[row, slot] = surface[chosen]
            self.valid[row, slot] = True
            self.count[row] += 1


def place_rigid(xy, margin, valid, center, half, accept, tries):
    """Yaw and offset that fit a rigid group on its surface; accept(candidates, pending, attempt) adds constraints."""
    n, device = len(xy), xy.device
    angle, offset = torch.zeros(n, device=device), torch.zeros(n, 2, device=device)
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    pending = torch.arange(n, device=device)
    for attempt in range(tries):
        if not len(pending):
            break
        draw = torch.rand(len(pending), device=device)*2*math.pi
        local = rotate_xy(xy[pending], draw[:, None])
        extent = margin[pending, :, None]
        missing = ~valid[pending, :, None]
        low = (local-extent).masked_fill(missing, torch.inf).amin(1)
        high = (local+extent).masked_fill(missing, -torch.inf).amax(1)
        free = half[pending]-(high-low)/2
        shift = center[pending]-(low+high)/2+(torch.rand_like(free)*2-1)*free.clamp_min(0)
        good = (free >= 0).all(-1) & accept(local+shift[:, None], pending, attempt)
        chosen = pending[good]
        angle[chosen], offset[chosen], placed[chosen] = draw[good], shift[good], True
        pending = pending[~good]
    return angle, offset, placed


def place_rigid_batched(xy, margin, valid, center, half, accept, tries):
    n, device = len(xy), xy.device
    angle, offset = torch.zeros(n, device=device), torch.zeros(n, 2, device=device)
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    gap = (xy[:, :, None]-xy[:, None]).norm(dim=-1)
    reach = 2*half[:, None, None]-margin[:, :, None, None]-margin[:, None, :, None]
    pairs = valid[:, :, None] & valid[:, None]
    fits = ~((gap > reach.clamp_min(0).norm(dim=-1)+1e-6) & pairs).any((1, 2))
    fits &= (~valid | (margin <= half.amin(-1, keepdim=True))).all(-1)
    rows = fits.nonzero().flatten()
    if not len(rows):
        return angle, offset, placed
    draw = torch.rand(len(rows), tries, 3, device=device)
    turns = draw[..., 0]*2*math.pi
    local = rotate_xy(xy[rows, None], turns[..., None])
    extent, missing = margin[rows, None, :, None], ~valid[rows, None, :, None]
    low = (local-extent).masked_fill(missing, torch.inf).amin(-2)
    high = (local+extent).masked_fill(missing, -torch.inf).amax(-2)
    free = half[rows, None]-(high-low)/2
    shifts = center[rows, None]-(low+high)/2+(draw[..., 1:]*2-1)*free.clamp_min(0)
    feasible = (free >= 0).all(-1)
    chosen, candidate = feasible.nonzero(as_tuple=True)
    good = torch.zeros_like(feasible)
    if len(chosen):
        good[chosen, candidate] = accept(local[chosen, candidate]+shifts[chosen, candidate, None], rows[chosen], 0)
    accepted, first = good.any(-1), good.long().argmax(-1)
    selected = torch.arange(len(rows), device=device)
    angle[rows] = torch.where(accepted, turns[selected, first], 0.)
    offset[rows] = torch.where(accepted[:, None], shifts[selected, first], 0.)
    placed[rows] = accepted
    return angle, offset, placed
