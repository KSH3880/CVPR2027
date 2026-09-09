"""Fixed V1 supervised objectives."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .geometry import path_to_shared
from .schema import SPEED_VALUES, PlannerState


LOSS_WEIGHTS = {"waypoint": 1.0, "speed": 0.5, "smoothness": 0.1, "collision": 0.5}


def _sample_coarse_at_arc(path: torch.Tensor, arc: torch.Tensor) -> torch.Tensor:
    seg = path[:, 1:] - path[:, :-1]
    length = seg.norm(dim=-1).clamp(min=1e-6)
    cum = torch.cat((torch.zeros_like(length[:, :1]), length.cumsum(dim=1)), dim=1)
    index = torch.searchsorted(cum.detach().contiguous(), arc.detach().contiguous(), right=True)
    index = (index - 1).clamp(0, path.shape[1] - 2)
    ar = torch.arange(path.shape[0], device=path.device)[:, None]
    start = cum.gather(1, index)
    frac = ((arc - start) / length.gather(1, index)).clamp(0, 1).unsqueeze(-1)
    return path[ar, index] + frac * (path[ar, index + 1] - path[ar, index])


def differentiable_timed_positions(
    path: torch.Tensor,
    speed_logits: torch.Tensor,
    steps: int = 64,
) -> torch.Tensor:
    """Differentiable expected-speed rollout used only by collision loss."""
    table = torch.tensor(SPEED_VALUES, device=path.device, dtype=path.dtype)
    speed = (speed_logits.softmax(dim=-1) * table).sum(dim=-1)
    seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1)
    end_s = seg.sum(dim=1)
    qlen = end_s[:, None] / 4.0
    duration = qlen / speed.clamp(min=1e-4)
    start_t = torch.cat((torch.zeros_like(duration[:, :1]), duration.cumsum(dim=1)[:, :-1]), dim=1)
    makespan = duration.sum(dim=1)
    # Common wall-clock samples keep the two agents time-aligned. Detaching the
    # grid range avoids an unstable gradient through a batch maximum.
    tmax = makespan.detach().max().clamp(min=1e-3)
    time = torch.linspace(0, 1, steps, device=path.device, dtype=path.dtype)[None] * tmax
    elapsed = (time[:, :, None] - start_t[:, None]).clamp(min=0.0)
    elapsed = torch.minimum(elapsed, duration[:, None])
    arc = (elapsed * speed[:, None]).sum(dim=-1).clamp(max=end_s[:, None])
    return _sample_coarse_at_arc(path, arc)


def collision_loss(coarse_world: torch.Tensor, speed_logits: torch.Tensor, clearance: float = 1.0) -> torch.Tensor:
    b = coarse_world.shape[0]
    path = coarse_world.reshape(b * 2, coarse_world.shape[2], 2)
    logits = speed_logits.reshape(b * 2, 4, 4)
    positions = differentiable_timed_positions(path, logits).reshape(b, 2, -1, 2)
    distance = (positions[:, 0] - positions[:, 1]).norm(dim=-1)
    return F.relu(clearance - distance).square().mean()


def compute_loss(
    output: Dict[str, torch.Tensor],
    state: PlannerState,
    target_path_world: torch.Tensor,
    target_speed: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    target_local = path_to_shared(target_path_world, state)
    point_mask = torch.ones_like(target_local[..., 0], dtype=torch.bool)
    carry = state.phase >= 0.5
    point_mask[:, :, 1:16] &= ~carry[:, :, None]
    point_error = F.smooth_l1_loss(output["coarse_local"], target_local, reduction="none").mean(dim=-1)
    waypoint = point_error[point_mask].mean()
    speed = F.cross_entropy(output["speed_logits"].reshape(-1, 4), target_speed.reshape(-1).long())

    second = output["coarse_local"][:, :, 2:] - 2 * output["coarse_local"][:, :, 1:-1] + output["coarse_local"][:, :, :-2]
    smooth_mask = torch.ones_like(second[..., 0], dtype=torch.bool)
    smooth_mask[:, :, 14:17] = False  # pickup corner is allowed to be discontinuous
    smooth_mask[:, :, :14] &= ~carry[:, :, None]
    smoothness = second.square().sum(dim=-1)[smooth_mask].mean()
    collision = collision_loss(output["coarse_world"], output["speed_logits"])
    total = (
        LOSS_WEIGHTS["waypoint"] * waypoint
        + LOSS_WEIGHTS["speed"] * speed
        + LOSS_WEIGHTS["smoothness"] * smoothness
        + LOSS_WEIGHTS["collision"] * collision
    )
    return {
        "total": total,
        "waypoint": waypoint,
        "speed": speed,
        "smoothness": smoothness,
        "collision": collision,
    }
