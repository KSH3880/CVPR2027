"""Goal offsets aligned with the box's local +Y, as in koo Push Stage1."""
import torch


def aligned_goal_offset(yaw,distance):
    return torch.stack((-distance*yaw.sin(),distance*yaw.cos(),torch.zeros_like(distance)),-1)
