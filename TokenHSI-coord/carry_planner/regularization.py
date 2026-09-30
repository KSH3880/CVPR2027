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


def _arc_lookahead_delta(
    path: torch.Tensor, root_xy: torch.Tensor, distance: float,
) -> torch.Tensor:
    # Return root-to-path vectors at a fixed detached arc-length lookup.
    batch, agents, points, coordinates = path.shape
    if coordinates != 2 or root_xy.shape != (batch, agents, 2):
        raise ValueError("lookahead path/root shape mismatch")
    segment_length = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    arc = torch.cat((
        torch.zeros_like(segment_length[..., :1]),
        segment_length.cumsum(dim=-1),
    ), dim=-1)
    arc_lookup = arc.detach().reshape(-1, points).contiguous()
    target_arc = torch.minimum(
        arc[..., -1].detach(), arc.new_full((), distance),
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
    point0 = path.gather(2, gather).squeeze(2)
    gather = upper[..., None, None].expand(-1, -1, 1, 2)
    point1 = path.gather(2, gather).squeeze(2)
    return point0 + blend[..., None] * (point1 - point0) - root_xy


def _sample_at_arc(path: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Sample [B,A,P,2] paths at [B,A,S] metric arc distances."""
    batch, agents, points, coordinates = path.shape
    if coordinates != 2 or distance.shape[:2] != (batch, agents):
        raise ValueError("arc sample path/distance shape mismatch")
    segment = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    arc = torch.cat((torch.zeros_like(segment[..., :1]), segment.cumsum(-1)), -1)
    lookup = arc.detach().reshape(-1, points).contiguous()
    target = distance.detach().reshape(-1, distance.shape[-1]).contiguous()
    upper = torch.searchsorted(lookup, target, right=False).clamp(1, points - 1)
    lower = upper - 1
    arc0 = lookup.gather(1, lower)
    arc1 = lookup.gather(1, upper)
    blend = ((target - arc0) / (arc1 - arc0).clamp(min=1e-7)).clamp(0.0, 1.0)
    flat = path.reshape(-1, points, coordinates)
    point0 = flat.gather(1, lower[..., None].expand(-1, -1, coordinates))
    point1 = flat.gather(1, upper[..., None].expand(-1, -1, coordinates))
    return (point0 + blend[..., None] * (point1 - point0)).reshape(
        batch, agents, distance.shape[-1], coordinates,
    )


def _suffix_consistency(path, observation, progress, beta, decay, cap,
                        safe_weight):
    """Compare unexecuted paths by traveled meters, not by point index."""
    points = path.shape[-2]
    root = observation.state.root_xy
    previous = observation.previous_path_world.detach()
    index = torch.arange(points, device=path.device, dtype=path.dtype)
    # Discard the executed prefix and re-anchor its remainder at the measured
    # root, including tracking error accumulated since the previous decision.
    executed = index.reshape(1, 1, points) <= progress.floor()[..., None]
    remaining = torch.where(executed[..., None], root[..., None, :], previous)
    old_length = (remaining[..., 1:, :] - remaining[..., :-1, :]).norm(dim=-1).sum(-1)
    new_length = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).sum(-1)
    common_length = torch.minimum(old_length, new_length).detach()
    fraction = torch.linspace(0.0, 1.0, points, device=path.device, dtype=path.dtype)
    distance = common_length[..., None] * fraction[None, None, :]
    old_sample = _sample_at_arc(remaining, distance)
    new_sample = _sample_at_arc(path, distance)
    displacement = (new_sample - old_sample).norm(dim=-1)
    spacing = old_length.detach() / (points - 1 - progress).clamp(min=1.0)
    decay_m = (spacing * decay).clamp(min=0.1)
    weight = torch.exp(-distance / decay_m[..., None])
    weight[..., 0] = 0.0  # both routes are anchored at the measured root
    per_agent_cost = torch.where(
        displacement < beta,
        0.5 * displacement.square() / beta,
        displacement - 0.5 * beta,
    )
    per_agent_cost = (per_agent_cost * weight).sum(-1) / weight.sum(-1).clamp(min=1e-7)
    per_agent_displacement = (displacement * weight).sum(-1) / weight.sum(-1).clamp(min=1e-7)
    active = (common_length > 0.1).to(path.dtype)
    per_sample_cost = (per_agent_cost * active).sum(-1) / active.sum(-1).clamp(min=1.0)
    per_sample_displacement = (
        (per_agent_displacement * active).sum(-1) / active.sum(-1).clamp(min=1.0)
    )
    eligible = observation.previous_path_valid.to(path.dtype) * active.any(-1).to(path.dtype)
    loss = _bounded_weighted_mean(per_sample_cost, eligible * safe_weight, cap)
    metric = _bounded_weighted_mean(per_sample_displacement, eligible, float("inf"))
    return loss, metric.detach()


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
    pickup_fallback_weight: float = 0.50,
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
    if (not math.isfinite(pickup_fallback_weight)
            or not 0.0 <= pickup_fallback_weight <= 1.0):
        raise ValueError("pickup fallback weight must be in [0, 1]")

    suffix_replan = bool(output.get("suffix_replan", False))
    progress = observation.path_progress
    if progress.shape != (batch, agents):
        raise ValueError("path_progress must be [B,2]")
    # A held agent has completed the pickup leg even if projection noise leaves
    # the previous-path cursor just before the fixed box anchor.
    effective_progress = (
        progress if suffix_replan else torch.where(
            observation.state.held >= 0.5,
            torch.maximum(progress, progress.new_full((), 16.0)),
            progress,
        )
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
    if suffix_replan:
        consistency_loss, replan_displacement = _suffix_consistency(
            path, observation, progress, consistency_beta,
            near_future_decay, loss_cap, safe_weight,
        )
    else:
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
        replan_displacement = _bounded_weighted_mean(
            displacement.mean(dim=(1, 2)),
            observation.previous_path_valid.to(path.dtype), float("inf"),
        ).detach()

    state = observation.state
    # Progress is measured on the previous path. Weighting a newly changed
    # spline by that old fractional index can omit the measured-root to first
    # future-point connection. Collapse the executed prefix onto the current
    # root, as the analytic rollout does, and measure the complete suffix.
    past = point_index.reshape(1, 1, points) <= effective_progress[..., None]
    future_path = (
        path if suffix_replan else torch.where(
            past[..., None], state.root_xy[..., None, :], path,
        )
    )
    segment_length = (
        future_path[..., 1:, :] - future_path[..., :-1, :]
    ).norm(dim=-1)
    future_length = segment_length.sum(dim=-1)

    planned_delta = _arc_lookahead_delta(
        future_path, state.root_xy, direction_lookahead,
    )
    planned_norm = planned_delta.norm(dim=-1)
    planned_direction = (
        planned_delta / planned_norm[..., None].clamp(min=1e-7)
    )
    root_speed = state.root_vel_xy.norm(dim=-1)
    velocity_direction = (
        state.root_vel_xy / root_speed[..., None].clamp(min=1e-7)
    )
    speed_weight = (
        (root_speed - direction_min_speed)
        / (direction_full_speed - direction_min_speed)
    ).clamp(0.0, 1.0)

    # At pickup the executor nearly stops, so velocity alone provides no
    # tangent reference. Reuse the previously committed carry direction and
    # blend smoothly back to physical velocity as the agent accelerates.
    previous_future_path = torch.where(
        past[..., None], state.root_xy[..., None, :],
        observation.previous_path_world.detach(),
    )
    previous_delta = _arc_lookahead_delta(
        previous_future_path, state.root_xy, direction_lookahead,
    )
    previous_norm = previous_delta.norm(dim=-1)
    previous_direction = (
        previous_delta / previous_norm[..., None].clamp(min=1e-7)
    )
    goal_remaining = (state.root_xy - state.goal_xy).norm(dim=-1)
    fallback_available = (
        (state.held >= 0.5)
        & observation.previous_path_valid[:, None]
        & (goal_remaining > direction_lookahead)
        & (previous_norm > 0.10)
    )
    fallback_strength = (
        pickup_fallback_weight * (1.0 - speed_weight)
        * fallback_available.to(path.dtype)
    )
    reference_vector = (
        speed_weight[..., None] * velocity_direction
        + fallback_strength[..., None] * previous_direction
    )
    reference_norm = reference_vector.norm(dim=-1)
    reference_direction = (
        reference_vector / reference_norm[..., None].clamp(min=1e-7)
    )
    direction_cosine = (
        planned_direction * reference_direction
    ).sum(dim=-1).clamp(-1.0, 1.0)
    direction_active = (reference_norm > 1e-6) & (planned_norm > 0.10)
    direction_weight = (speed_weight + fallback_strength).clamp(max=1.0)
    free_cosine = math.cos(math.radians(direction_free_angle_deg))
    direction_error = torch.relu(free_cosine - direction_cosine).square()
    direction_loss = (
        (direction_error * direction_weight
         * direction_active.to(path.dtype)).sum()
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
        "mean_replan_displacement": replan_displacement,
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
        "direction_fallback_fraction": fallback_available.float().mean().detach(),
    }


__all__ = ["carry_path_regularization"]
