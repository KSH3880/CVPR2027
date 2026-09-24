"""Small bounded geometry terms for recurrent plain-Carry replanning."""

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
    free_detour_ratio: float = 1.20,
    consistency_beta: float = 0.25,
    near_future_decay: float = 6.0,
    safety_gate_scale: float = 0.05,
    loss_cap: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Stabilize safe replans and charge only excessive future path length.

    Both objectives are gated by detached analytic collision risk.  A risky
    proposal therefore remains free to change route; once it is safe, the
    planner is weakly encouraged to retain its near future and remove only
    detours beyond ``free_detour_ratio``.
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
    for name, value in (
        ("consistency_beta", consistency_beta),
        ("near_future_decay", near_future_decay),
        ("safety_gate_scale", safety_gate_scale),
        ("loss_cap", loss_cap),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    progress = observation.path_progress
    if progress.shape != (batch, agents):
        raise ValueError("path_progress must be [B,2]")
    safe_weight = torch.exp(
        -collision_risk.detach().clamp(min=0.0) / safety_gate_scale
    )

    point_index = torch.arange(
        points, device=path.device, dtype=path.dtype,
    )
    ahead = point_index.reshape(1, 1, points) - progress[..., None]
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

    segment_length = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    segment_index = torch.arange(
        points - 1, device=path.device, dtype=path.dtype,
    )
    remaining_fraction = (
        segment_index.reshape(1, 1, points - 1) + 1.0
        - progress[..., None]
    ).clamp(0.0, 1.0)
    future_length = (segment_length * remaining_fraction).sum(dim=-1)
    state = observation.state
    box_xy = state.box_xyz[..., :2]
    direct_approach = (
        (state.root_xy - box_xy).norm(dim=-1)
        + (box_xy - state.goal_xy).norm(dim=-1)
    )
    direct_carry = (state.root_xy - state.goal_xy).norm(dim=-1)
    direct = torch.where(state.held >= 0.5, direct_carry, direct_approach)
    active_agent = direct > 0.10
    length_ratio = future_length / direct.clamp(min=0.10)
    excess = torch.relu(length_ratio - free_detour_ratio).square()
    per_sample_length = (
        (excess * active_agent.to(path.dtype)).sum(dim=1)
        / active_agent.to(path.dtype).sum(dim=1).clamp(min=1.0)
    )
    length_eligible = active_agent.any(dim=1).to(path.dtype) * safe_weight
    excess_length_loss = _bounded_weighted_mean(
        per_sample_length, length_eligible, loss_cap,
    )

    return {
        "consistency_loss": consistency_loss,
        "excess_length_loss": excess_length_loss,
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
    }


__all__ = ["carry_path_regularization"]
