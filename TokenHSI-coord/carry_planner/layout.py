"""Pure-torch geometry for Carry planner stress layouts."""

from __future__ import annotations

import torch


def converging_goal_xy(box_xy, crossing, box_size_xy, margin,
                       minimum_direction_separation=0.25):
    """Put close goals beyond a shared crossing on both box rays.

    Every feasible carry segment is ``box -> crossing -> goal``. Equal ray
    distance after the crossing makes goal separation exactly the conservative
    box-radius clearance. Nearly parallel incoming rays are rejected instead
    of producing an arbitrarily distant endpoint.
    """
    if box_xy.ndim != 3 or box_xy.shape[1:] != (2, 2):
        raise ValueError("box_xy must be [B,2,2]")
    if crossing.shape != (box_xy.shape[0], 2):
        raise ValueError("crossing must be [B,2]")
    if box_size_xy.shape != box_xy.shape:
        raise ValueError("box_size_xy must be [B,2,2]")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if minimum_direction_separation <= 0.0:
        raise ValueError("minimum_direction_separation must be positive")
    toward = crossing[:, None] - box_xy
    distance = toward.norm(dim=-1)
    direction = toward / distance[..., None].clamp(min=1e-6)
    direction_separation = (direction[:, 0] - direction[:, 1]).norm(dim=-1)
    feasible = (
        (distance > 1e-4).all(dim=-1)
        & (direction_separation >= float(minimum_direction_separation))
    )
    radius = 0.5 * box_size_xy.norm(dim=-1)
    separation = radius.sum(dim=-1) + float(margin)
    post = separation / direction_separation.clamp(
        min=float(minimum_direction_separation)
    )
    goal = crossing[:, None] + post[:, None, None] * direction
    return goal, feasible


__all__ = ["converging_goal_xy"]
