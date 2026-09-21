"""Pure-torch geometry for Carry planner stress layouts."""

from __future__ import annotations

import torch


def converging_goal_xy(center, box_size_xy, margin, angle):
    """Place two feasible goals tightly around a common center.

    Circumscribed box radii make the endpoint conservative for arbitrary box
    yaw.  The random separation axis prevents the planner from memorizing a
    fixed left/right yielding convention.
    """
    if center.ndim != 2 or center.shape[-1] != 2:
        raise ValueError("center must be [B,2]")
    if box_size_xy.shape != (center.shape[0], 2, 2):
        raise ValueError("box_size_xy must be [B,2,2]")
    if angle.shape != (center.shape[0],):
        raise ValueError("angle must be [B]")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    radius = 0.5 * box_size_xy.norm(dim=-1)
    separation = radius.sum(dim=-1) + float(margin)
    direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    offset = 0.5 * separation[:, None] * direction
    return torch.stack((center - offset, center + offset), dim=1)


__all__ = ["converging_goal_xy"]
