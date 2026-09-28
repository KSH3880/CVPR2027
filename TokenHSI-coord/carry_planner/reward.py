"""Small reward terms specific to the plain Carry planner."""

from __future__ import annotations

import math

import torch


def apply_invalid_plan_penalty(
    reward: torch.Tensor, valid: torch.Tensor, coefficient: float,
    *, max_turn_deg: torch.Tensor = None, turn_limit_deg: float = 46.0,
):
    """Charge rejected proposals, with extra cost for severe curve violations.

    ``coefficient`` remains the minimum cost of every rejected proposal. A turn
    above the executable limit increases that cost linearly up to 2x at twice
    the limit. This preserves fallback-credit protection while giving PPO an
    ordering among rejected noisy samples.
    """
    if reward.ndim != 1 or valid.shape != reward.shape or valid.dtype != torch.bool:
        raise ValueError("reward and bool valid must be matching [B] tensors")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("invalid-plan coefficient must be finite and non-negative")
    if not math.isfinite(turn_limit_deg) or turn_limit_deg <= 0.0:
        raise ValueError("turn limit must be finite and positive")
    multiplier = torch.ones_like(reward)
    if max_turn_deg is not None:
        if max_turn_deg.shape != reward.shape:
            raise ValueError("max_turn_deg must match reward shape")
        relative_excess = max_turn_deg.to(reward.dtype) / turn_limit_deg - 1.0
        relative_excess = torch.nan_to_num(
            relative_excess, nan=1.0, posinf=1.0, neginf=0.0,
        ).clamp(0.0, 1.0)
        multiplier = multiplier + relative_excess
    penalty = (~valid).to(reward.dtype) * coefficient * multiplier
    return reward - penalty, penalty


__all__ = ["apply_invalid_plan_penalty"]
