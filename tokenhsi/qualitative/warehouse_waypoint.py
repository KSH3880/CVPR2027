import argparse
import math

import torch

from utils.before_object_alias import alias_layout, encoder_packet, goal_aligned_scene


ONTOP_TEMPLATE = 4
ONTOP_PACKET_TASK = 3
ONTOP_WAYPOINT_RADIUS = 1.2


def switch_margin(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or not (value * 2).is_integer():
        raise argparse.ArgumentTypeError('switch margin must be nonnegative, in 0.5 m increments')
    return value


def door_box(origin, shift, yaw, room_half, thickness, floor_height, payload_size):
    local_x = -room_half[:, 0] - thickness / 2
    result = origin.clone()
    result[:, 0] += shift[:, 0] + local_x * yaw.cos()
    result[:, 1] += shift[:, 1] + local_x * yaw.sin()
    result[:, 2] += floor_height + payload_size[:, 2] / 2
    return result


def update_switch(root_xy, virtual_xy, payload_size, margin, switched, *, fixed_radius=None):
    distance = (root_xy - virtual_xy).norm(dim=-1)
    radius = (payload_size[:, :2].norm(dim=-1) / 2 + margin
              if fixed_radius is None else distance.new_full(distance.shape, fixed_radius))
    return switched | (distance <= radius), distance, radius


def update_return(holding, payload_xy, virtual_xy, payload_size, margin, picked, returned, *,
                  fixed_radius=None):
    picked = picked | holding
    reached, distance, radius = update_switch(payload_xy, virtual_xy, payload_size, margin, returned,
                                             fixed_radius=fixed_radius)
    return picked, returned | (picked & reached), distance, radius


def waypoint_scene(scene, leader, virtual_local, num_agents=4, num_objects=8, target=None):
    # Resolve the follower's alias against the real final goal before changing either pose.
    result = goal_aligned_scene(scene, num_agents, num_objects)
    packet = result[:, -5*num_agents:].reshape(len(scene), -1, 5).long()
    record = ((packet[..., 0] == 1) & (packet[..., 1] == 2)
              & (packet[..., 2] == leader[:, None])).long().argmax(-1)
    rows = torch.arange(len(scene), device=scene.device)
    payload = packet[rows, record, 3]
    if target is not None:
        payload = torch.where(target, packet[rows, record, 4], payload)
    end = result.shape[1] - 5*num_agents
    poses = result[:, end-7*(2*num_agents+num_objects):end].reshape(len(scene), -1, 7)
    poses[rows, payload, :3] = virtual_local
    return result


def ontop_waypoint_scene(scene, follower, virtual_local, num_agents=4, num_objects=8, target=None):
    """Move an ON_TOP follower's payload or goal alias in GTA."""
    result = goal_aligned_scene(scene, num_agents, num_objects)
    end = result.shape[1] - 5*num_agents
    packet = result[:, end:]
    dependent, _, alias, _ = alias_layout(packet, num_agents, num_objects)
    _, task, actor, payload, _ = packet.reshape(len(scene), -1, 5).long().unbind(-1)
    rows, records = (dependent & (task == ONTOP_PACKET_TASK) &
                     (actor == follower[:, None])).nonzero(as_tuple=True)
    slot = payload[rows, records]
    if target is not None:
        slot = torch.where(target[rows], alias[rows, records], slot)
    poses = result[:, end-7*(2*num_agents+num_objects):end].reshape(len(scene), -1, 7)
    poses[rows, slot, :3] = virtual_local[rows]
    return result


def role_waypoint_inputs(scene, people, phases, door, num_agents, num_objects, *, box_only=False):
    """Build one batch of the real view and a separate view per active role."""
    width = 5*num_agents
    packet = scene[:, -width:]
    layout = alias_layout(packet, num_agents, num_objects)
    base = goal_aligned_scene(scene, num_agents, num_objects, layout=layout)
    redirected = encoder_packet(packet, num_agents, num_objects, layout=layout)
    active = phases == 0
    if not box_only:
        active |= phases == 2
    rows, roles = active.nonzero(as_tuple=True)
    people = people[rows, roles]
    if not len(rows):
        return torch.cat((base[:, :-width], redirected), -1), packet, rows, people

    dependent, _, aliases, _ = layout
    valid, task, actor, payload, target = packet[rows].reshape(len(rows), -1, 5).long().unbind(-1)
    follower = roles == 1
    expected = torch.where(follower, ONTOP_PACKET_TASK, 2)
    match = (valid.bool() & (task == expected[:, None]) & (actor == people[:, None])
             & (~follower[:, None] | dependent[rows]))
    record = match.long().argmax(-1, keepdim=True)
    slots = payload
    if not box_only:
        goal = torch.where(follower[:, None], aliases[rows], target)
        slots = torch.where((phases[rows, roles] == 2)[:, None], goal, payload)
    found = match.any(-1, keepdim=True)
    slots = slots.gather(1, record).masked_fill(~found, 0).squeeze(1)

    guided = base[rows, :-width]
    pose_width = 7*(2*num_agents+num_objects)
    poses = guided[:, -pose_width:].reshape(len(rows), -1, 7)
    view = torch.arange(len(rows), device=scene.device)
    poses[view, slots, :3] = torch.where(found, door[rows], poses[view, slots, :3])
    inputs = torch.cat((torch.cat((base[:, :-width], guided)), torch.cat((redirected, redirected[rows]))), -1)
    return inputs, torch.cat((packet, packet[rows])), rows, people


def worker_spawns(roots, objects, sizes, valid, leader, follower, source, goal,
                  follower_task, near, candidates=32):
    n, m = roots.shape[:2]
    rows = torch.arange(n, device=roots.device)
    result = roots.clone()
    assigned = torch.ones(n, m, device=roots.device, dtype=torch.bool)
    assigned[rows, leader] = False
    assigned[rows, follower] = False
    success = torch.ones(n, device=roots.device, dtype=torch.bool)
    for leading, person in ((True, leader), (False, follower)):
        points = roots.new_empty(n, candidates, 2).uniform_(0, 1)
        if leading:
            points[..., 0] = -3.5 + points[..., 0]
            side = torch.where(source[rows, leader, 1] >= 0, 1., -1.)
            points[..., 1] = side[:, None] * (2.4 + .8*points[..., 1])
        else:
            points[..., 0] = -3.5 + 2.6*points[..., 0]
            points[..., 1] = -3.6 + 7.2*points[..., 1]
        target = source[rows, person] if leading else torch.where(
            (follower_task < 3)[:, None], goal, source[rows, follower])
        distance = (points - target[:, None]).norm(dim=-1)
        limit = 5. if leading else torch.where(follower_task < 3, 10., 5.)[:, None]
        good = (distance >= 1.) & (distance <= limit)
        if not leading:
            first = ((result[rows, leader]-source[rows, leader]).norm(dim=-1)
                     + (source[rows, leader]-goal).norm(dim=-1))
            carry = (source[rows, follower]-goal).norm(dim=-1)
            route = distance + torch.where(follower_task == 4, carry, 0.)[:, None]
            good &= ~near[:, None] | (route < first[:, None]-.001)
            good &= (points-goal[:, None]).norm(dim=-1) >= 1.001
        separation = (points[:, :, None]-result[:, None]).norm(dim=-1)
        good &= ~((separation < .85) & assigned[:, None]).any(-1)
        gap = ((points[:, :, None]-objects[:, None]).abs()-sizes[:, None]/2).clamp_min(0).norm(dim=-1)
        good &= ~((gap < .5) & valid[:, None]).any(-1)
        found = good.any(-1)
        result[rows, person] = points[rows, good.long().argmax(-1)]
        assigned[rows, person] = found
        success &= found
    return result, success
