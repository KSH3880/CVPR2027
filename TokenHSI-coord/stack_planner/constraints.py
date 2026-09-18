"""Differentiable route constraints for the free stack-planner paths."""

from __future__ import annotations

from typing import Dict

import torch

from coordinator.schema import MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES

from .schema import AGENTS, STACK_PATH_POINTS


SOFT_TURN_LIMIT_DEG = 46.0
HARD_TURN_LIMIT_DEG = 175.0


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


def retreat_box_clearance(
    path: torch.Tensor,
    box_xy: torch.Tensor,
    box_yaw: torch.Tensor,
    box_size_xy: torch.Tensor,
    *,
    agent_radius: float = 0.35,
    safety_margin: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Penalize a retreat path that enters the placed box's safety footprint.

    The oriented box is expanded by the agent root radius and a safety margin.
    A1 may initially be inside that expanded region immediately after release,
    so the loss grows with path progress and separately penalizes any step that
    moves deeper toward the box.  Moving monotonically outward is permitted.
    """
    if path.ndim < 3 or path.shape[-1] != 2 or path.shape[-2] < 2:
        raise ValueError("path must end in [P,2] with P >= 2")
    leading = path.shape[:-2]
    if box_xy.shape != leading + (2,):
        raise ValueError("box_xy must match path leading dimensions")
    if box_yaw.shape != leading or box_size_xy.shape != leading + (2,):
        raise ValueError("box yaw/size must match path leading dimensions")
    if agent_radius < 0.0 or safety_margin < 0.0:
        raise ValueError("agent radius and safety margin must be non-negative")
    if (box_size_xy <= 0.0).any():
        raise ValueError("box_size_xy must be positive")

    delta = path - box_xy.unsqueeze(-2)
    cosine = torch.cos(box_yaw).unsqueeze(-1)
    sine = torch.sin(box_yaw).unsqueeze(-1)
    local = torch.stack((
        cosine * delta[..., 0] + sine * delta[..., 1],
        -sine * delta[..., 0] + cosine * delta[..., 1],
    ), dim=-1)
    half_extent = (
        0.5 * box_size_xy + agent_radius + safety_margin
    ).unsqueeze(-2)
    q = local.abs() - half_extent
    # Standard signed distance to an axis-aligned rectangle in box coordinates.
    signed_distance = (
        q.clamp(min=0.0).norm(dim=-1)
        + q.amax(dim=-1).clamp(max=0.0)
    )
    progress = torch.linspace(
        0.0, 1.0, path.shape[-2], device=path.device, dtype=path.dtype,
    )
    inside = (-signed_distance).clamp(min=0.0)
    lingering = (progress * inside.square()).mean(dim=-1)
    inward_step = (
        signed_distance[..., :-1] - signed_distance[..., 1:]
    ).clamp(min=0.0)
    inward = inward_step.square().mean(dim=-1)
    # A path that merely exits and then ends back inside the footprint is not
    # a usable retreat. This is collision clearance, not a fixed-distance
    # goal: there is no reward for moving farther once the endpoint is outside.
    endpoint_overlap = (-signed_distance[..., -1]).clamp(min=0.0).square()
    return {
        "penalty": lingering + inward + endpoint_overlap,
        "endpoint_overlap": endpoint_overlap,
        "minimum_clearance": signed_distance.amin(dim=-1),
        "endpoint_clearance": signed_distance[..., -1],
    }


def retreat_endpoint_change_cost(current, previous, mask, tolerance=0.10):
    """Soft, bounded world-space goal drift cost, never a goal latch."""
    drift = (current - previous).norm(dim=-1)
    return (drift - tolerance).clamp(min=0.0, max=2.0).square() * mask.float()


def held_box_body_cost(body_xyz, box_xyz, box_yaw, box_size, held,
                       body_radius=0.08, margin=0.15):
    """Other-agent non-hand body proximity to an oriented held box (3D)."""
    delta = body_xyz - box_xyz[:, None, :]
    c, s = torch.cos(box_yaw)[:, None], torch.sin(box_yaw)[:, None]
    local = torch.stack((c * delta[..., 0] + s * delta[..., 1],
                         -s * delta[..., 0] + c * delta[..., 1],
                         delta[..., 2]), dim=-1)
    q = local.abs() - box_size[:, None, :] / 2
    sdf = q.clamp(min=0).norm(dim=-1) + q.amax(-1).clamp(max=0)
    return (body_radius + margin - sdf).clamp(min=0).square().amax(-1) * held.float()


def free_path_validity(
    path: torch.Tensor, speed: torch.Tensor, root_xy: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Validate fixed-origin unified paths near the current executing root."""
    return free_path_validity_details(path, speed, root_xy, active)["valid"]


def free_path_validity_details(
    path: torch.Tensor, speed: torch.Tensor, root_xy: torch.Tensor,
    active: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return the complete validity decision and each hard-gate component."""
    expected = (AGENTS, STACK_PATH_POINTS, 2)
    if path.shape != (*speed.shape, 2) or path.shape[2:] != expected:
        raise ValueError(
            f"expected [B,K,{AGENTS},{STACK_PATH_POINTS},2] path and matching speed"
        )
    finite = (
        torch.isfinite(path).flatten(start_dim=-2).all(dim=-1)
        & torch.isfinite(speed).all(dim=-1)
    )
    root_projection = project_points_to_segments(
        path, root_xy[:, None].expand(-1, path.shape[1], -1, -1),
    )
    # After the first decision P0 is the immutable departure point. The live
    # root therefore only has to remain reachable from the full polyline; the
    # executor resumes at its projected arc instead of demanding P0 == root.
    root_reachable = root_projection["distance2"].amin(dim=-1) < 1.0
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
    max_turn = turns.amax(dim=-1)
    turn_excess_cost = (
        ((turns - SOFT_TURN_LIMIT_DEG).clamp(min=0.0) / 90.0).square()
    ).mean(dim=-1)
    agent_valid = (
        finite & root_reachable & buffer_ok & speed_ok
        & (max_turn <= HARD_TURN_LIMIT_DEG)
    )
    return {
        "valid": (agent_valid | ~active[:, None]).all(dim=-1),
        "agent_valid": agent_valid,
        "finite": finite,
        "root_reachable": root_reachable,
        "buffer_ok": buffer_ok,
        "speed_ok": speed_ok,
        "max_turn_deg": max_turn,
        "turn_excess_cost": turn_excess_cost,
        "path_length": lengths.sum(dim=-1),
    }


__all__ = [
    "free_path_validity", "free_path_validity_details",
    "ordered_box_goal_visit", "retreat_box_clearance",
    "project_points_to_segments", "SOFT_TURN_LIMIT_DEG",
    "HARD_TURN_LIMIT_DEG",
]
