"""Geometry terms for dynamically feasible plain-Carry replanning."""

from __future__ import annotations

import math
from typing import Dict

import torch

from stack_planner.history import StackPlannerObservation


def _bounded_weighted_mean(
    value: torch.Tensor, weight: torch.Tensor, cap: float,
) -> torch.Tensor:
    numerator = (value * weight).sum()
    denominator = weight.sum().clamp(min=1.0)
    mean = numerator / denominator
    if math.isinf(cap):
        return mean
    # Saturate the contribution without the zero-gradient region introduced
    # by a hard clamp.  Large detours still receive a small corrective signal.
    return cap * mean / (cap + mean)


def carry_path_regularization(
    output: Dict[str, torch.Tensor],
    observation: StackPlannerObservation,
    collision_risk: torch.Tensor,
    *,
    free_detour_ratio: float = 1.15,
    absolute_length_slack: float = 0.25,
    length_huber_beta: float = 0.50,
    direction_lookahead: float = 0.50,
    direction_free_angle_deg: float = 15.0,
    direction_min_speed: float = 0.20,
    direction_full_speed: float = 0.80,
    consistency_beta: float = 0.25,
    near_future_decay: float = 6.0,
    safety_gate_scale: float = 0.05,
    loss_cap: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Stabilize safe replans and charge only excessive future path length.

    Consistency and length are gated by detached analytic collision risk.
    Direction alignment remains active because an instantaneous turn away
    from physical velocity is infeasible even during collision avoidance.
    """
    path = output["path_world"]
    if path.ndim != 5 or path.shape[1] != 1:
        raise ValueError("carry regularization expects path [B,1,2,33,2]")
    path = path[:, 0]
    batch, agents, points, _ = path.shape
    if collision_risk.shape != (batch,):
        raise ValueError("collision_risk must be [B]")
    if free_detour_ratio < 1.0:
        raise ValueError("free_detour_ratio must be at least one")
    if not math.isfinite(absolute_length_slack) or absolute_length_slack < 0.0:
        raise ValueError("absolute_length_slack must be finite and non-negative")
    for name, value in (
        ("length_huber_beta", length_huber_beta),
        ("direction_lookahead", direction_lookahead),
        ("direction_full_speed", direction_full_speed),
        ("consistency_beta", consistency_beta),
        ("near_future_decay", near_future_decay),
        ("safety_gate_scale", safety_gate_scale),
        ("loss_cap", loss_cap),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (not math.isfinite(direction_min_speed)
            or direction_min_speed < 0.0
            or direction_full_speed <= direction_min_speed):
        raise ValueError("direction speed range must be finite and increasing")
    if (not math.isfinite(direction_free_angle_deg)
            or not 0.0 <= direction_free_angle_deg < 180.0):
        raise ValueError("direction free angle must be in [0, 180)")

    progress = observation.path_progress
    if progress.shape != (batch, agents):
        raise ValueError("path_progress must be [B,2]")
    # A held agent has completed the pickup leg even if projection noise leaves
    # the previous-path cursor just before the fixed box anchor.
    effective_progress = torch.where(
        observation.state.held >= 0.5,
        torch.maximum(progress, progress.new_full((), 16.0)),
        progress,
    )
    safe_weight = torch.exp(
        -collision_risk.detach().clamp(min=0.0) / safety_gate_scale
    )

    point_index = torch.arange(
        points, device=path.device, dtype=path.dtype,
    )
    ahead = point_index.reshape(1, 1, points) - effective_progress[..., None]
    future_weight = torch.exp(
        -ahead.clamp(min=0.0) / near_future_decay
    ) * (ahead > 0.0).to(path.dtype)
    future_weight = future_weight * observation.previous_path_valid[
        :, None, None
    ].to(path.dtype)
    displacement = (
        path - observation.previous_path_world.detach()
    ).norm(dim=-1)
    huber = torch.where(
        displacement < consistency_beta,
        0.5 * displacement.square() / consistency_beta,
        displacement - 0.5 * consistency_beta,
    )
    per_sample_consistency = (
        (huber * future_weight).sum(dim=(1, 2))
        / future_weight.sum(dim=(1, 2)).clamp(min=1.0)
    )
    consistency_eligible = (
        observation.previous_path_valid.to(path.dtype) * safe_weight
    )
    consistency_loss = _bounded_weighted_mean(
        per_sample_consistency, consistency_eligible, loss_cap,
    )

    state = observation.state
    # Progress is measured on the previous path. Weighting a newly changed
    # spline by that old fractional index can omit the measured-root to first
    # future-point connection. Collapse the executed prefix onto the current
    # root, as the analytic rollout does, and measure the complete suffix.
    past = point_index.reshape(1, 1, points) <= effective_progress[..., None]
    future_path = torch.where(
        past[..., None], state.root_xy[..., None, :], path,
    )
    segment_length = (
        future_path[..., 1:, :] - future_path[..., :-1, :]
    ).norm(dim=-1)
    future_length = segment_length.sum(dim=-1)

    # Measure the outgoing path direction at a fixed arc-length lookahead.
    # Lookup distances are detached so length changes cannot game selection;
    # gradients still reach the selected/interpolated path points.
    arc = torch.cat((
        torch.zeros_like(segment_length[..., :1]),
        segment_length.cumsum(dim=-1),
    ), dim=-1)
    arc_lookup = arc.detach().reshape(-1, points).contiguous()
    target_arc = torch.minimum(
        future_length.detach(),
        future_length.new_full((), direction_lookahead),
    ).reshape(-1, 1)
    upper = torch.searchsorted(arc_lookup, target_arc, right=False).squeeze(-1)
    upper = upper.clamp(1, points - 1).reshape(batch, agents)
    lower = upper - 1
    arc0 = arc_lookup.gather(1, lower.reshape(-1, 1)).reshape(batch, agents)
    arc1 = arc_lookup.gather(1, upper.reshape(-1, 1)).reshape(batch, agents)
    blend = (
        (target_arc.reshape(batch, agents) - arc0)
        / (arc1 - arc0).clamp(min=1e-7)
    ).clamp(0.0, 1.0)
    gather = lower[..., None, None].expand(-1, -1, 1, 2)
    point0 = future_path.gather(2, gather).squeeze(2)
    gather = upper[..., None, None].expand(-1, -1, 1, 2)
    point1 = future_path.gather(2, gather).squeeze(2)
    lookahead_point = point0 + blend[..., None] * (point1 - point0)

    planned_delta = lookahead_point - state.root_xy
    planned_norm = planned_delta.norm(dim=-1)
    planned_direction = (
        planned_delta / planned_norm[..., None].clamp(min=1e-7)
    )
    root_speed = state.root_vel_xy.norm(dim=-1)
    actual_direction = (
        state.root_vel_xy / root_speed[..., None].clamp(min=1e-7)
    )
    direction_cosine = (
        planned_direction * actual_direction
    ).sum(dim=-1).clamp(-1.0, 1.0)
    speed_weight = (
        (root_speed - direction_min_speed)
        / (direction_full_speed - direction_min_speed)
    ).clamp(0.0, 1.0)
    direction_active = (
        (root_speed > direction_min_speed) & (planned_norm > 0.10)
    )
    free_cosine = math.cos(math.radians(direction_free_angle_deg))
    direction_error = torch.relu(free_cosine - direction_cosine).square()
    direction_loss = (
        (direction_error * speed_weight * direction_active.to(path.dtype)).sum()
        / direction_active.to(path.dtype).sum().clamp(min=1.0)
    )

    box_xy = state.box_xyz[..., :2]
    direct_approach = (
        (state.root_xy - box_xy).norm(dim=-1)
        + (box_xy - state.goal_xy).norm(dim=-1)
    )
    direct_carry = (state.root_xy - state.goal_xy).norm(dim=-1)
    direct = torch.where(state.held >= 0.5, direct_carry, direct_approach)
    active_agent = direct > 0.10
    length_ratio = future_length / direct.clamp(min=0.10)
    allowed_length = free_detour_ratio * direct + absolute_length_slack
    excess_m = torch.relu(future_length - allowed_length)
    # A quadratic basin keeps small numerical/spline deviations gentle, while
    # the linear tail preserves a non-vanishing gradient for severe detours.
    excess = torch.where(
        excess_m < length_huber_beta,
        0.5 * excess_m.square() / length_huber_beta,
        excess_m - 0.5 * length_huber_beta,
    )
    per_sample_length = (
        (excess * active_agent.to(path.dtype)).sum(dim=1)
        / active_agent.to(path.dtype).sum(dim=1).clamp(min=1.0)
    )
    length_eligible = active_agent.any(dim=1).to(path.dtype) * safe_weight
    excess_length_loss = _bounded_weighted_mean(
        per_sample_length, length_eligible, float("inf"),
    )

    return {
        "consistency_loss": consistency_loss,
        "excess_length_loss": excess_length_loss,
        "direction_loss": direction_loss,
        "safe_weight": safe_weight.mean().detach(),
        "mean_replan_displacement": _bounded_weighted_mean(
            displacement.mean(dim=(1, 2)),
            observation.previous_path_valid.to(path.dtype),
            float("inf"),
        ).detach(),
        "mean_future_length_ratio": (
            (length_ratio * active_agent.to(path.dtype)).sum()
            / active_agent.to(path.dtype).sum().clamp(min=1.0)
        ).detach(),
        "mean_future_excess_m": (
            (excess_m * active_agent.to(path.dtype)).sum()
            / active_agent.to(path.dtype).sum().clamp(min=1.0)
        ).detach(),
        "max_future_excess_m": torch.where(
            active_agent, excess_m, torch.zeros_like(excess_m),
        ).max().detach(),
        "mean_direction_error_deg": (
            torch.rad2deg(torch.acos(direction_cosine.detach()))
            * direction_active.to(path.dtype)
        ).sum() / direction_active.to(path.dtype).sum().clamp(min=1.0),
        "direction_active_fraction": direction_active.float().mean().detach(),
    }


__all__ = ["carry_path_regularization"]
