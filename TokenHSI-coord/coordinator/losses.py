"""Auxiliary C1 objectives; no hand-written trajectory labels are required."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .planner import candidate_costs
from .schema import MAX_SPEED, CoordinatorState


def compute_auxiliary_loss(output: Dict[str, torch.Tensor], state: CoordinatorState) -> Dict[str, torch.Tensor]:
    metrics = candidate_costs(output, state)
    # Soft best-of-K lets useful modes specialize without assigning semantic names.
    best_of_k = -0.2 * torch.logsumexp(-metrics["train_cost"] / 0.2, dim=1).mean()
    speed_nominal = (MAX_SPEED - output["speed"]).square().mean()
    accel_smooth = (
        output["acceleration"][..., 1:] - output["acceleration"][..., :-1]
    ).square().mean()
    residual = output["control_residual"].flatten(start_dim=2)
    distance = torch.cdist(residual, residual)
    eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)[None]
    diversity = F.relu(0.5 - distance.masked_fill(eye, 1e6)).square().sum()
    diversity = diversity / max(state.batch_size * 12, 1)
    risk_target = torch.stack(
        (
            metrics["min_hh"] < 1.0,
            metrics["min_bb_margin"] < 0.0,
            metrics["min_hb_margin"] < 0.0,
        ),
        dim=-1,
    ).to(output["risk_logits"].dtype)
    risk = F.binary_cross_entropy_with_logits(output["risk_logits"], risk_target.detach())
    total = best_of_k + 0.05 * speed_nominal + 0.05 * accel_smooth + 0.1 * diversity + 0.1 * risk
    return {
        "total": total,
        "best_of_k": best_of_k,
        "speed_nominal": speed_nominal,
        "accel_smooth": accel_smooth,
        "diversity": diversity,
        "risk": risk,
    }
