import torch

from utils.coupled_push_assignment import assign_regions
from utils.coupled_push_spec import assign_regions as sample_regions, region_states


def restore_training_assignment(task, ids):
    boxes = task._assigned_box_values(task._box_states, ids)
    sizes = task._assigned_box_values(task._box_size, ids)
    delta = task._humanoid_root_states[ids, :, :2]-boxes[..., :2]
    yaw = task._structure_yaw[ids, 1, None]
    local_x = yaw.cos()*delta[..., 0]+yaw.sin()*delta[..., 1]
    pushers = task._bridge_pushers[ids]
    previous = task._regions[ids].clone()
    # Initial Bridge placement assigns region 0/1 along the structure's x axis.
    task._regions[ids] = torch.where(pushers, (local_x > 0).long(), previous)
    tokens, _ = region_states(boxes, sizes, task._members[ids], task._regions[ids])
    offset = tokens[..., :3]-boxes[..., :3]
    task._tar_pos[ids] = torch.where(pushers[..., None],
        task._center_tar_pos[ids]+offset, task._tar_pos[ids])
    return int(((previous != task._regions[ids]) & pushers).any(-1).sum())


def restore_ordered_assignment(task, ids):
    boxes = task._assigned_box_values(task._box_states, ids)
    sizes = task._assigned_box_values(task._box_size, ids)
    pushers = task._bridge_pushers[ids]
    previous = task._regions[ids].clone()
    regions = assign_regions(task._groups[ids], task._humanoid_root_states[ids],
        boxes, sizes, previous, 'ordered', 0.)
    task._regions[ids] = torch.where(pushers, regions, previous)
    tokens, _ = region_states(boxes, sizes, task._members[ids], task._regions[ids])
    offset = tokens[..., :3]-boxes[..., :3]
    task._tar_pos[ids] = torch.where(pushers[..., None],
        task._center_tar_pos[ids]+offset, task._tar_pos[ids])
    return int(((previous != task._regions[ids]) & pushers).any(-1).sum())


def randomize_pusher_assignment(task, ids):
    pushers = task._bridge_pushers[ids]
    regions = sample_regions(task._groups[ids])
    task._regions[ids] = torch.where(pushers, regions, task._regions[ids])
    boxes = task._assigned_box_values(task._box_states, ids)
    sizes = task._assigned_box_values(task._box_size, ids)
    tokens, _ = region_states(boxes, sizes, task._members[ids], task._regions[ids])
    offset = tokens[..., :3]-boxes[..., :3]
    task._tar_pos[ids] = torch.where(pushers[..., None],
        task._center_tar_pos[ids]+offset, task._tar_pos[ids])
