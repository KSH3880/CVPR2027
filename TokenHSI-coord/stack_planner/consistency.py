"""Temporal plan-consistency objective for receding-horizon stack planning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from coordinator.schema import MAX_SPEED, CoordinatorState


CONSISTENCY_START_S = 0.5
CONSISTENCY_STEP_S = 0.25
CONSISTENCY_SAMPLES = 32


def _query(reference: torch.Tensor) -> torch.Tensor:
    return CONSISTENCY_START_S + CONSISTENCY_STEP_S * torch.arange(
        CONSISTENCY_SAMPLES, device=reference.device, dtype=reference.dtype,
    )


def _sample_polyline(
    path: torch.Tensor, speed: torch.Tensor, query_seconds: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Sample a no-dwell path at absolute seconds from its start."""
    if path.shape[:-1] != speed.shape or path.shape[-1] != 2:
        raise ValueError("path/speed shape mismatch")
    if query_seconds.ndim != 1 or query_seconds.numel() == 0:
        raise ValueError("query_seconds must be a non-empty 1-D tensor")
    if (query_seconds < 0).any():
        raise ValueError("query_seconds must be non-negative")

    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    average_speed = 0.5 * (speed[..., 1:] + speed[..., :-1]).clamp(min=1e-4)
    arrival = torch.cat((
        torch.zeros_like(ds[..., :1]),
        (ds / average_speed).cumsum(dim=-1),
    ), dim=-1)
    query = query_seconds.to(path).reshape(1, 1, 1, -1).expand(
        path.shape[0], path.shape[1], path.shape[2], -1,
    )
    flat_arrival = arrival.reshape(-1, arrival.shape[-1]).contiguous()
    flat_query = query.reshape(-1, query.shape[-1]).contiguous()
    index = torch.searchsorted(flat_arrival, flat_query, right=True)
    index = index.sub(1).clamp(0, path.shape[-2] - 2)
    row = torch.arange(flat_arrival.shape[0], device=path.device)[:, None]
    t0 = flat_arrival.gather(1, index)
    t1 = flat_arrival.gather(1, index + 1)
    fraction = ((flat_query - t0) / (t1 - t0).clamp(min=1e-5)).clamp(0, 1)

    flat_path = path.reshape(-1, path.shape[-2], 2)
    p0, p1 = flat_path[row, index], flat_path[row, index + 1]
    position = p0 + fraction[..., None] * (p1 - p0)
    flat_speed = speed.reshape(-1, speed.shape[-1])
    v0, v1 = flat_speed.gather(1, index), flat_speed.gather(1, index + 1)
    sampled_speed = v0 + fraction * (v1 - v0)
    shape = (*query.shape, 2)
    return {
        "position": position.reshape(shape),
        "speed": sampled_speed.reshape(query.shape),
        "valid": query <= arrival[..., -1, None],
    }


@torch.no_grad()
def build_stack_consistency_target(
    previous_output: Dict[str, torch.Tensor],
    previous_state: CoordinatorState,
    current_state: CoordinatorState,
    elapsed_seconds: float,
    active_env: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Advance the preceding mean plan and detach its unexecuted future."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if active_env.shape != (current_state.batch_size,):
        raise ValueError("active_env must be [B]")
    query = _query(previous_output["path_world"]) + elapsed_seconds
    previous_speed = torch.full(
        previous_output["path_world"].shape[:-1], MAX_SPEED,
        device=previous_output["path_world"].device,
        dtype=previous_output["path_world"].dtype,
    )
    sampled = _sample_polyline(
        previous_output["path_world"], previous_speed, query,
    )
    same_phase = current_state.phase == previous_state.phase
    base_valid = active_env[:, None, None, None].bool() & same_phase[:, None, :, None]
    return {
        "position": sampled["position"].detach(),
        "valid": (sampled["valid"] & base_valid).detach(),
    }


def stack_trajectory_consistency_loss(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    target: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Penalize changes from the time-aligned remainder of the prior plan."""
    zero = output["path_world"].new_zeros(())
    if target is None:
        return {
            "total": zero, "position": zero,
            "valid_fraction": zero,
        }
    query = _query(output["path_world"])
    nominal_speed = torch.full(
        output["path_world"].shape[:-1], MAX_SPEED,
        device=output["path_world"].device, dtype=output["path_world"].dtype,
    )
    sampled = _sample_polyline(output["path_world"], nominal_speed, query)
    mask = sampled["valid"] & target["valid"].bool()
    weight = mask.to(output["path_world"].dtype)
    valid_count = weight.sum()
    position_error = F.smooth_l1_loss(
        sampled["position"], target["position"], reduction="none", beta=0.25,
    ).mean(dim=-1)
    position_sum = (weight * position_error).sum()
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
