"""Temporal plan consistency for fixed-index stack trajectories."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from coordinator.schema import CoordinatorState


@torch.no_grad()
def build_stack_consistency_target(
    previous_output: Dict[str, torch.Tensor],
    previous_state: CoordinatorState,
    current_state: CoordinatorState,
    elapsed_seconds: float,
    active_env: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Keep the preceding fixed-index plan as the temporal target."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if active_env.shape != (current_state.batch_size,):
        raise ValueError("active_env must be [B]")
    same_phase = current_state.phase == previous_state.phase
    points = previous_output["path_world"].shape[-2]
    base_valid = (
        active_env[:, None, None, None].bool()
        & same_phase[:, None, :, None]
    ).expand(-1, previous_output["path_world"].shape[1], -1, points)
    return {
        "position": previous_output["path_world"].detach().clone(),
        # expand() creates a stride-0 view. Training further intersects this
        # mask in place with the planner-decision mask, so materialize it.
        "valid": base_valid.detach().clone(),
    }


def stack_trajectory_consistency_loss(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    target: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Penalize changes at matching fixed path indices."""
    zero = output["path_world"].new_zeros(())
    if target is None:
        return {
            "total": zero, "position": zero,
            "valid_fraction": zero,
        }
    if output["path_world"].shape != target["position"].shape:
        raise ValueError("fixed-index consistency target shape mismatch")
    mask = target["valid"].bool()
    weight = mask.to(output["path_world"].dtype)
    valid_count = weight.sum()
    position_error = F.smooth_l1_loss(
        output["path_world"], target["position"], reduction="none", beta=0.25,
    ).mean(dim=-1)
    position_sum = (weight * position_error * target.get("position_scale", 1.0)).sum()
    denominator = valid_count.clamp(min=1.0)
    position = position_sum / denominator
    return {
        "total": position,
        "position": position,
        "valid_fraction": valid_count / weight.new_tensor(weight.numel()).clamp(min=1.0),
    }


__all__ = [
    "build_stack_consistency_target", "stack_trajectory_consistency_loss",
]
