"""Differentiable full-path collision supervision for plain Carry."""

from __future__ import annotations

from typing import Dict

import torch

from coordinator.planner import candidate_costs
from coordinator.schema import CoordinatorState


def carry_analytic_collision_loss(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    initial_plan: torch.Tensor,
    *,
    focus_steps: int = 8,
    human_clearance: float = 1.0,
    box_margin: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Apply 96-sample/top-k collision loss to fresh episode plans.

    Recurrent carry paths retain their already executed prefix. Running a full
    normalized rollout on later replans would incorrectly treat that past
    prefix as future motion, so the analytic target is restricted to the first
    full plan of each episode. Speed determines arrival timing but is detached:
    this auxiliary objective explicitly teaches spatial avoidance, while the
    physical PPO reward remains free to learn dynamic speed control.
    """
    path = output["path_world"]
    speed = output["speed"]
    if path.ndim != 5 or path.shape[1] != 1:
        raise ValueError("carry analytic loss expects path [B,1,2,33,2]")
    if speed.shape != path.shape[:-1]:
        raise ValueError("carry analytic loss speed shape must match path points")
    if initial_plan.shape != (path.shape[0],):
        raise ValueError("initial_plan must be bool [B]")
    if focus_steps < 1 or focus_steps > 96:
        raise ValueError("focus_steps must be in [1, 96]")
    if human_clearance <= 0.0 or box_margin < 0.0:
        raise ValueError("clearance must be positive and margin non-negative")

    active = initial_plan.bool()
    zero = path.sum() * 0.0
    if not bool(active.any()):
        return {
            "loss": zero,
            "active_fraction": active.float().mean(),
            "min_hh": zero.detach(),
            "min_bb_margin": zero.detach(),
            "min_hb_margin": zero.detach(),
        }

    selected = {
        "path_world": path[active],
        # Timing stays faithful to the current policy, but collision gradient
        # cannot take the easier speed-only escape route.
        "speed": speed[active].detach(),
        "pickup_dwell": speed.new_zeros(
            int(active.sum()), path.shape[1], path.shape[2]
        ),
    }
    metrics = candidate_costs(
        selected,
        state.index(active),
        human_clearance=human_clearance,
        box_margin=box_margin,
        measured_executor_timing=True,
    )
    focused = metrics["future_collision_steps"].topk(
        focus_steps, dim=-1
    ).values.mean()
    return {
        "loss": focused,
        "active_fraction": active.float().mean(),
        "min_hh": metrics["min_hh"].mean().detach(),
        "min_bb_margin": metrics["min_bb_margin"].mean().detach(),
        "min_hb_margin": metrics["min_hb_margin"].mean().detach(),
    }


__all__ = ["carry_analytic_collision_loss"]
