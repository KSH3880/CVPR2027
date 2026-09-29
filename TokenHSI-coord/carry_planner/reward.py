"""Small reward terms specific to the plain Carry planner."""

from __future__ import annotations

import math

import torch

from coordinator.schema import CoordinatorState


def carry_remaining_distance(state: CoordinatorState) -> torch.Tensor:
    """Mean remaining root-to-box-to-goal distance across both agents.

    Keep the goal leg in the potential before pickup so lifting the box does
    not create an artificial negative progress spike. Phase 3 is a grounded
    box at its goal, so its remaining distance is zero after placement.
    """
    box_xy = state.box_xyz[..., :2]
    approach = (state.root_xy - box_xy).norm(dim=-1)
    carry = (box_xy - state.goal_xy).norm(dim=-1)
    remaining = carry + torch.where(
        state.held >= 0.5, torch.zeros_like(approach), approach,
    )
    remaining = torch.where(
        state.phase >= 2.5, torch.zeros_like(remaining), remaining,
    )
    return remaining.mean(dim=1)


def apply_invalid_plan_penalty(
    reward: torch.Tensor, valid: torch.Tensor, coefficient: float,
):
    """Charge each rejected sampled proposal without masking its PPO credit."""
    if reward.ndim != 1 or valid.shape != reward.shape or valid.dtype != torch.bool:
        raise ValueError("reward and bool valid must be matching [B] tensors")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("invalid-plan coefficient must be finite and non-negative")
    penalty = (~valid).to(reward.dtype) * coefficient
    return reward - penalty, penalty


__all__ = ["carry_remaining_distance", "apply_invalid_plan_penalty"]
