"""Small reward terms specific to the plain Carry planner."""

from __future__ import annotations

import math

import torch


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


__all__ = ["apply_invalid_plan_penalty"]
