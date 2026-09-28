"""Pure diagnostics for plain-Carry hard plan validation."""

from __future__ import annotations

from typing import Dict

import torch

from coordinator.schema import (
    MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES, CoordinatorState,
)


def carry_plan_validity_debug(
    state: CoordinatorState, path: torch.Tensor, speed: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Mirror Carry curvature checks and expose degenerate-segment rejects."""
    if path.ndim != 4 or path.shape[-2:] != (33, 2):
        raise ValueError("path must be [B,2,33,2]")
    if speed.shape != path.shape[:-1]:
        raise ValueError("speed must be [B,2,33]")
    if state.root_xy.shape != path.shape[:2] + (2,):
        raise ValueError("state/path batch or agent shape mismatch")

    finite = torch.isfinite(path).flatten(start_dim=1).all(dim=-1)
    finite &= torch.isfinite(speed).flatten(start_dim=1).all(dim=-1)
    segment = path[..., 1:, :] - path[..., :-1, :]
    segment_length = segment.norm(dim=-1)
    buffer_ok = (
        segment_length.sum(dim=-1) < (PATH_VERTICES - 2) * PATH_DS
    ).all(dim=1)
    speed_ok = (
        (speed >= MIN_SPEED - 1e-5) & (speed <= MAX_SPEED + 1e-5)
    ).flatten(start_dim=1).all(dim=-1)

    v0, v1 = segment[..., :-1, :], segment[..., 1:, :]
    product = v0.norm(dim=-1) * v1.norm(dim=-1)
    cosine = (v0 * v1).sum(dim=-1) / product.clamp(min=1e-7)
    turn = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    ignored = torch.zeros_like(turn, dtype=torch.bool)
    ignored[..., 14:17] = True
    ignored[..., :16] |= state.held[..., None] >= 0.5
    evaluated_turn = torch.where(ignored, torch.zeros_like(turn), turn)
    curve_ok = evaluated_turn.flatten(start_dim=1).amax(dim=-1) <= 46.0

    degenerate = product <= 1e-10
    zero_safe_turn = torch.where(
        ignored | degenerate, torch.zeros_like(turn), turn,
    )
    zero_safe_curve_ok = (
        zero_safe_turn.flatten(start_dim=1).amax(dim=-1) <= 46.0
    )
    relevant_degenerate = (degenerate & ~ignored).flatten(start_dim=1).any(dim=-1)
    return {
        "finite": finite,
        "buffer": buffer_ok,
        "speed": speed_ok,
        "curve": curve_ok,
        "zero_safe_curve": zero_safe_curve_ok,
        "zero_turn_false_reject": ~curve_ok & zero_safe_curve_ok,
        "relevant_degenerate_turn": relevant_degenerate,
        "max_turn_deg": evaluated_turn.flatten(start_dim=1).amax(dim=-1),
        "path_length_m": segment_length.sum(dim=-1).mean(dim=1),
    }


__all__ = ["carry_plan_validity_debug"]
