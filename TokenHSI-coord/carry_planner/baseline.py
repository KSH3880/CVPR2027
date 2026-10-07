"""Handcrafted Carry routes without learned proposals or avoidance."""
import torch


def straight_carry_path(root_xy, box_xy, goal_xy, held):
    """33 points: root→box→goal before pickup, root→goal while held."""
    t = torch.linspace(0, 1, 17, device=root_xy.device, dtype=root_xy.dtype)
    first = root_xy[..., None, :] + t[:, None] * (box_xy - root_xy)[..., None, :]
    second = box_xy[..., None, :] + t[:, None] * (goal_xy - box_xy)[..., None, :]
    unheld = torch.cat((first, second[..., 1:, :]), dim=-2)
    t = torch.linspace(0, 1, 33, device=root_xy.device, dtype=root_xy.dtype)
    direct = root_xy[..., None, :] + t[:, None] * (goal_xy - root_xy)[..., None, :]
    path = torch.where((held >= 0.5)[..., None, None], direct, unheld)
    return path, torch.full_like(path[..., 0], 1.5)
