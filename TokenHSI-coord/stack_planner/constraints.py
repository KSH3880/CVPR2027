"""Differentiable route constraints for the free stack-planner paths."""

from __future__ import annotations

from typing import Dict

import torch

from coordinator.schema import MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES

from .schema import AGENTS, STACK_PATH_POINTS


def project_points_to_segments(
    path: torch.Tensor, points: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Project one point per path onto every finite polyline segment.

    ``path`` is ``[..., P, 2]`` and ``points`` is ``[..., 2]``.  The returned
    tensors have one value (or foot point) per segment.
    """
    if path.ndim < 3 or path.shape[-1] != 2:
        raise ValueError("path must end in [P,2]")
    if points.shape != path.shape[:-2] + (2,):
        raise ValueError("points must match path leading dimensions")
    start, delta = path[..., :-1, :], path[..., 1:, :] - path[..., :-1, :]
    relative = points.unsqueeze(-2) - start
    fraction = (
        (relative * delta).sum(dim=-1)
        / delta.square().sum(dim=-1).clamp(min=1e-8)
    ).clamp(0.0, 1.0)
    foot = start + fraction.unsqueeze(-1) * delta
    distance2 = (points.unsqueeze(-2) - foot).square().sum(dim=-1)
    return {"foot": foot, "fraction": fraction, "distance2": distance2}


def ordered_box_goal_visit(
    path: torch.Tensor,
    box_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    tolerance: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Require box then goal visitation without assigning waypoint indices.

    The minimizing pair may use any two segments.  If both points project onto
    the same segment, their projection fractions must still be ordered.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    expected = path.shape[:-2] + (2,)
    if box_xy.shape != expected or goal_xy.shape != expected:
        raise ValueError("box_xy and goal_xy must match path leading dimensions")
    box = project_points_to_segments(path, box_xy)
    goal = project_points_to_segments(path, goal_xy)
    segments = path.shape[-2] - 1
    index = torch.arange(segments, device=path.device)
    box_index = index.reshape((1,) * (path.ndim - 2) + (segments, 1))
    goal_index = index.reshape((1,) * (path.ndim - 2) + (1, segments))
    ordered = goal_index > box_index
    same_segment = goal_index == box_index
    fraction_ordered = (
        goal["fraction"].unsqueeze(-2) >= box["fraction"].unsqueeze(-1)
    )
    allowed = ordered | (same_segment & fraction_ordered)
    pair_cost = (
        box["distance2"].unsqueeze(-1)
        + goal["distance2"].unsqueeze(-2)
    )
    pair_cost = pair_cost.masked_fill(~allowed, float("inf"))
    flat = pair_cost.flatten(start_dim=-2)
    choice = flat.argmin(dim=-1)
    box_choice = torch.div(choice, segments, rounding_mode="floor")
    goal_choice = choice.remainder(segments)
    box_d2 = box["distance2"].gather(-1, box_choice.unsqueeze(-1)).squeeze(-1)
    goal_d2 = goal["distance2"].gather(-1, goal_choice.unsqueeze(-1)).squeeze(-1)
    box_distance = box_d2.clamp(min=0).sqrt()
    goal_distance = goal_d2.clamp(min=0).sqrt()
    penalty = (
        (box_distance - tolerance).clamp(min=0).square()
        + (goal_distance - tolerance).clamp(min=0).square()
    )
    return {
        "penalty": penalty,
        "box_distance": box_distance,
        "goal_distance": goal_distance,
        "box_segment": box_choice,
        "goal_segment": goal_choice,
    }


def free_path_validity(
    path: torch.Tensor, speed: torch.Tensor, root_xy: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Validate unified paths with only the current-root hard anchor."""
    expected = (AGENTS, STACK_PATH_POINTS, 2)
    if path.shape != (*speed.shape, 2) or path.shape[2:] != expected:
        raise ValueError(
            f"expected [B,K,{AGENTS},{STACK_PATH_POINTS},2] path and matching speed"
        )
    finite = (
        torch.isfinite(path).flatten(start_dim=-2).all(dim=-1)
        & torch.isfinite(speed).all(dim=-1)
    )
    root_anchor = (path[..., 0, :] - root_xy[:, None]).norm(dim=-1) < 0.01
    delta = path[..., 1:, :] - path[..., :-1, :]
    lengths = delta.norm(dim=-1)
    buffer_ok = lengths.sum(dim=-1) < (PATH_VERTICES - 2) * PATH_DS
    product = lengths[..., :-1] * lengths[..., 1:]
    cosine = (
        (delta[..., :-1, :] * delta[..., 1:, :]).sum(dim=-1)
        / product.clamp(min=1e-8)
    )
    turns = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    turns = torch.where(product > 1e-10, turns, torch.zeros_like(turns))
    # Preserve the three-cubic executor prior while its free junctions learn.
    turns[..., 8:11] = 0
    turns[..., 19:22] = 0
    speed_ok = ((speed >= MIN_SPEED) & (speed <= MAX_SPEED)).all(dim=-1)
    valid = finite & root_anchor & buffer_ok & speed_ok & (turns.amax(dim=-1) <= 46)
    return (valid | ~active[:, None]).all(dim=-1)


__all__ = [
    "free_path_validity", "ordered_box_goal_visit",
    "project_points_to_segments",
]
