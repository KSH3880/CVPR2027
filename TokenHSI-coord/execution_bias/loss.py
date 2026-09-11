"""Losses for deterministic execution-bias regression."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def curvature_weight(plan_xyv: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Up-weight turns using one minus the cosine of adjacent segments."""
    xy = plan_xyv[..., :2]
    left = xy[..., 1:-1, :] - xy[..., :-2, :]
    right = xy[..., 2:, :] - xy[..., 1:-1, :]
    denom = left.norm(dim=-1) * right.norm(dim=-1)
    cosine = (left * right).sum(dim=-1) / denom.clamp(min=1e-8)
    bend = torch.where(denom > 1e-8, 1.0 - cosine.clamp(-1.0, 1.0),
                       torch.zeros_like(cosine))
    middle = 1.0 + coefficient * bend
    edge = torch.ones_like(middle[..., :1])
    return torch.cat((edge, middle, edge), dim=-1)


def execution_bias_loss(
    predicted_error: torch.Tensor,
    target_error: torch.Tensor,
    plan_xyv: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    curvature_coefficient: float = 4.0,
    shape_coefficient: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Masked curve-aware Smooth-L1 loss for exact XY residuals."""
    if predicted_error.shape != target_error.shape:
        raise ValueError("predicted_error and target_error must have identical shape")
    if predicted_error.shape != plan_xyv.shape[:-1] + (2,):
        raise ValueError("error tensors must match plan [B,A,P,2]")
    if valid_mask is None:
        valid_mask = torch.ones_like(plan_xyv[..., 0], dtype=torch.bool)
    if valid_mask.shape != plan_xyv.shape[:-1]:
        raise ValueError("valid_mask must be [B,A,P]")
    valid_mask = valid_mask.bool()

    weights = curvature_weight(plan_xyv, curvature_coefficient) * valid_mask
    point_raw = F.smooth_l1_loss(
        predicted_error, target_error, reduction="none"
    ).mean(dim=-1)
    point = (point_raw * weights).sum() / weights.sum().clamp(min=1.0)

    pair_valid = valid_mask[..., 1:] & valid_mask[..., :-1]
    predicted_difference = predicted_error[..., 1:, :] - predicted_error[..., :-1, :]
    target_difference = target_error[..., 1:, :] - target_error[..., :-1, :]
    shape_raw = F.smooth_l1_loss(
        predicted_difference, target_difference, reduction="none"
    ).mean(dim=-1)
    shape = (shape_raw * pair_valid).sum() / pair_valid.sum().clamp(min=1)
    total = point + shape_coefficient * shape
    return {"total": total, "point": point, "shape": shape}
