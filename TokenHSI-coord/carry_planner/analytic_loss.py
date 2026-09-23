"""Differentiable full-path collision supervision for plain Carry."""

from __future__ import annotations

from typing import Dict

import math

import torch

from coordinator.planner import _arrival_times, _sample_at_time
from coordinator.schema import CoordinatorState


def carry_analytic_collision_loss(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    path_progress: torch.Tensor,
    *,
    focus_steps: int = 8,
    human_clearance: float = 1.0,
    box_margin: float = 0.15,
    time_uncertainty: float = 1.5,
    time_samples: int = 7,
) -> Dict[str, torch.Tensor]:
    """Apply robust 96-sample collision loss to every recurrent replan.

    The fixed episode path contains an already executed prefix.  Collapse that
    prefix onto the measured root before predicting future motion, then test a
    small linear set of relative executor timing offsets.  Speed determines
    arrival timing but is detached so this objective teaches spatial avoidance;
    physical PPO remains responsible for learning dynamic speed control.
    """
    path = output["path_world"]
    speed = output["speed"]
    if path.ndim != 5 or path.shape[1] != 1:
        raise ValueError("carry analytic loss expects path [B,1,2,33,2]")
    if speed.shape != path.shape[:-1]:
        raise ValueError("carry analytic loss speed shape must match path points")
    if path_progress.shape != (path.shape[0], path.shape[2]):
        raise ValueError("path_progress must be [B,2]")
    if focus_steps < 1 or focus_steps > 96:
        raise ValueError("focus_steps must be in [1, 96]")
    if human_clearance <= 0.0 or box_margin < 0.0:
        raise ValueError("clearance must be positive and margin non-negative")
    if not math.isfinite(time_uncertainty) or time_uncertainty < 0.0:
        raise ValueError("time_uncertainty must be finite and non-negative")
    if time_samples < 1 or (time_samples > 1 and time_samples % 2 == 0):
        raise ValueError("time_samples must be one or an odd positive integer")

    # A held agent has necessarily passed the pickup anchor even if projection
    # noise leaves its progress slightly below point 16.
    progress = torch.where(
        state.held >= 0.5,
        torch.maximum(path_progress, path_progress.new_full((), 16.0)),
        path_progress,
    )
    point_index = torch.arange(
        path.shape[-2], device=path.device, dtype=path.dtype,
    )
    past = point_index.reshape(1, 1, 1, -1) <= progress[:, None, :, None]
    root = state.root_xy[:, None, :, None, :]
    future_path = torch.where(past[..., None], root, path)

    # Timing stays faithful to the policy, but collision gradient cannot take
    # the easier speed-only escape route.
    detached_speed = speed.detach()
    dwell = speed.new_zeros(path.shape[:3])
    points, arrival = _arrival_times(
        future_path, detached_speed, dwell, state=state,
        measured_executor_timing=True,
    )
    duration = arrival[..., -1]
    tmax = duration.amax(dim=-1)
    unit = torch.linspace(0.0, 1.0, 96, device=path.device, dtype=path.dtype)
    query = tmax[..., None] * unit.reshape(1, 1, -1)
    offsets = (
        path.new_zeros(1)
        if time_samples == 1 else torch.linspace(
            -time_uncertainty, time_uncertainty, time_samples,
            device=path.device, dtype=path.dtype,
        )
    )
    box_radius = 0.5 * state.box_size_xy.norm(dim=-1)
    bb_limit = box_radius.sum(dim=-1)[:, None, None] + box_margin
    hb01_limit = 0.35 + box_radius[:, 1, None, None]
    hb10_limit = 0.35 + box_radius[:, 0, None, None]
    pickup_time = arrival[..., 17]
    robust_steps = path.new_zeros(path.shape[0], path.shape[1], 96)
    min_hh = path.new_full(path.shape[:2], float("inf"))
    min_bb_margin = path.new_full(path.shape[:2], float("inf"))
    min_hb_margin = path.new_full(path.shape[:2], float("inf"))
    for offset in offsets:
        time0 = query
        time1 = query + offset
        q0 = time0[:, :, None, :]
        q1 = time1.clamp(min=0.0)[:, :, None, :]
        root0 = _sample_at_time(points[:, :, 0:1], arrival[:, :, 0:1], q0)
        root1 = _sample_at_time(points[:, :, 1:2], arrival[:, :, 1:2], q1)
        carry0 = ((state.held[:, None, 0:1, None] >= 0.5)
                  | (q0 >= pickup_time[:, :, 0:1, None]))
        carry1 = ((state.held[:, None, 1:2, None] >= 0.5)
                  | (q1 >= pickup_time[:, :, 1:2, None]))
        box0 = torch.where(
            carry0[..., None], root0,
            state.box_xyz[:, None, 0:1, None, :2],
        )
        box1 = torch.where(
            carry1[..., None], root1,
            state.box_xyz[:, None, 1:2, None, :2],
        )
        hh = (root0 - root1).norm(dim=-1).squeeze(2)
        bb = (box0 - box1).norm(dim=-1).squeeze(2)
        hb01 = (root0 - box1).norm(dim=-1).squeeze(2)
        hb10 = (box0 - root1).norm(dim=-1).squeeze(2)
        collision = (
            torch.relu(human_clearance - hh).square()
            + torch.relu(bb_limit - bb).square()
            + 0.5 * (
                torch.relu(hb01_limit - hb01).square()
                + torch.relu(hb10_limit - hb10).square()
            )
        )
        # Negative shifted time refers to motion before the current replan.
        valid_time = time1 >= 0.0
        collision = collision * valid_time.to(path.dtype)
        robust_steps = torch.maximum(robust_steps, collision)
        inf = torch.full_like(hh, float("inf"))
        min_hh = torch.minimum(
            min_hh, torch.where(valid_time, hh, inf).amin(dim=-1),
        )
        min_bb_margin = torch.minimum(
            min_bb_margin,
            torch.where(valid_time, bb - bb_limit, inf).amin(dim=-1),
        )
        min_hb_margin = torch.minimum(
            min_hb_margin,
            torch.minimum(
                torch.where(
                    valid_time, hb01 - hb01_limit, inf,
                ).amin(dim=-1),
                torch.where(
                    valid_time, hb10 - hb10_limit, inf,
                ).amin(dim=-1),
            ),
        )

    focused = robust_steps.topk(focus_steps, dim=-1).values.mean()
    segment = future_path[..., 1:, :] - future_path[..., :-1, :]
    before, after = segment[..., :-1, :], segment[..., 1:, :]
    cosine = (before * after).sum(dim=-1) / (
        before.norm(dim=-1) * after.norm(dim=-1)
    ).clamp(min=1e-7)
    # Pickup is an intentional approach/carry corner and is exempt from the
    # executor's curvature check as well.
    cosine[..., 14:17] = 1.0
    cosine_limit = math.cos(math.radians(46.0))
    turn_index = torch.arange(
        1, path.shape[-2] - 1, device=path.device, dtype=path.dtype,
    )
    future_turn = turn_index.reshape(1, 1, 1, -1) > progress[:, None, :, None]
    curvature_violation = torch.relu(cosine_limit - cosine).square()
    curvature_violation = torch.where(
        future_turn, curvature_violation, torch.zeros_like(curvature_violation),
    )
    curvature = curvature_violation.flatten(start_dim=2).amax(dim=-1).mean()
    return {
        "loss": focused,
        "curvature_loss": curvature,
        "active_fraction": path.new_ones(()),
        "min_hh": min_hh.mean().detach(),
        "min_bb_margin": min_bb_margin.mean().detach(),
        "min_hb_margin": min_hb_margin.mean().detach(),
    }


__all__ = ["carry_analytic_collision_loss"]
