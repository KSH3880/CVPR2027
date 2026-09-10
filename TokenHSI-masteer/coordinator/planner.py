"""Deterministic safety/efficiency selection over learned joint candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .executor_calibration import (
    MS18_TIMING_APPROACH_SPEEDS,
    MS18_TIMING_CARRY_SPEEDS,
    MS18_TIMING_COMMAND_SPEEDS,
    MS18_TIMING_PHASE1_REMAINING_DWELL_S,
    MS18_TIMING_PICKUP_DWELL_S,
)
from .schema import AGENTS, MAX_SPEED, PATH_POINTS, CoordinatorState


@dataclass
class CandidateSelection:
    index: torch.Tensor
    path_world: torch.Tensor
    speed: torch.Tensor
    pickup_dwell: torch.Tensor
    valid: torch.Tensor
    safe: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def apply_fixed_priority(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    priority_agent: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Fix one random-per-episode agent to the straight, full-speed prior.

    The coordinator still predicts both counterfactual yield trajectories from
    physical state only.  The episode-owned priority index selects which
    prediction is ignored; no role or scenario token is added to the model.
    """
    path = output["path_world"]
    speed = output["speed"]
    if path.shape[:3] != speed.shape[:3] or path.shape[-2] != speed.shape[-1]:
        raise ValueError("path/speed shape mismatch")
    if path.shape[0] != state.batch_size or path.shape[2] != AGENTS:
        raise ValueError("priority constraint expects [B,K,2,P] trajectories")
    if priority_agent.shape != (state.batch_size,):
        raise ValueError("priority_agent must be [B]")
    if priority_agent.dtype == torch.bool or priority_agent.is_floating_point():
        raise TypeError("priority_agent must have an integer dtype")
    if ((priority_agent < 0) | (priority_agent >= AGENTS)).any():
        raise ValueError("priority_agent values must be 0 or 1")

    unit = torch.linspace(
        0.0, 1.0, 17, device=path.device, dtype=path.dtype
    )
    root = state.root_xy
    box = state.box_xyz[..., :2]
    goal = state.goal_xy
    approach = root[:, :, None] + unit[None, None, :, None] * (
        box - root
    )[:, :, None]
    carry = box[:, :, None] + unit[None, None, :, None] * (
        goal - box
    )[:, :, None]
    straight = torch.cat((approach, carry[:, :, 1:]), dim=-2)[:, None]
    straight = straight.expand(-1, path.shape[1], -1, -1, -1)

    agent_ids = torch.arange(AGENTS, device=path.device)
    priority = (agent_ids[None] == priority_agent[:, None])[:, None, :, None]
    constrained_path = torch.where(priority[..., None], straight, path)
    constrained_speed = torch.where(
        priority, torch.full_like(speed, MAX_SPEED), speed
    )
    ds = (
        constrained_path[..., 1:, :] - constrained_path[..., :-1, :]
    ).norm(dim=-1).clamp(min=1e-5)
    acceleration = (
        constrained_speed[..., 1:].square()
        - constrained_speed[..., :-1].square()
    ) / (2.0 * ds)

    constrained = dict(output)
    constrained["path_world"] = constrained_path
    constrained["speed"] = constrained_speed
    constrained["acceleration"] = acceleration
    constrained["trajectory"] = torch.cat(
        (constrained_path, constrained_speed.unsqueeze(-1)), dim=-1
    )
    residual_mask = priority[..., None]
    if "waypoint_residual" in constrained:
        constrained["waypoint_residual"] = torch.where(
            residual_mask,
            torch.zeros_like(constrained["waypoint_residual"]),
            constrained["waypoint_residual"],
        )
    return constrained


def _piecewise_actual_speed(
    commanded: torch.Tensor, actual_values: tuple[float, ...]
) -> torch.Tensor:
    """Differentiable piecewise-linear ms18 command-to-motion lookup."""
    knots = commanded.new_tensor(MS18_TIMING_COMMAND_SPEEDS)
    actual = commanded.new_tensor(actual_values)
    value = commanded.clamp(min=float(knots[0]), max=float(knots[-1]))
    upper = torch.searchsorted(knots, value.contiguous(), right=True).clamp(
        min=1, max=knots.numel() - 1
    )
    lower = upper - 1
    x0, x1 = knots[lower], knots[upper]
    y0, y1 = actual[lower], actual[upper]
    fraction = (value - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def _measured_pickup_dwell(state: CoordinatorState, candidates: int) -> torch.Tensor:
    phase = state.phase[:, None].expand(-1, candidates, -1)
    dwell = torch.where(
        phase < 0.5,
        torch.full_like(phase, MS18_TIMING_PICKUP_DWELL_S),
        torch.where(
            phase < 1.5,
            torch.full_like(phase, MS18_TIMING_PHASE1_REMAINING_DWELL_S),
            torch.zeros_like(phase),
        ),
    )
    return torch.where(
        state.held[:, None] >= 0.5, torch.zeros_like(dwell), dwell
    )


def _arrival_times(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    *,
    state: CoordinatorState | None = None,
    measured_executor_timing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    avg_speed = 0.5 * (speed[..., 1:] + speed[..., :-1]).clamp(min=1e-4)
    if measured_executor_timing:
        if state is None:
            raise ValueError("state is required for measured executor timing")
        approach = _piecewise_actual_speed(
            avg_speed[..., :16], MS18_TIMING_APPROACH_SPEEDS
        )
        carry = _piecewise_actual_speed(
            avg_speed[..., 16:], MS18_TIMING_CARRY_SPEEDS
        )
        avg_speed = torch.cat((approach, carry), dim=-1)
        dwell = _measured_pickup_dwell(state, path.shape[1])
    base = torch.cat((torch.zeros_like(ds[..., :1]), (ds / avg_speed).cumsum(dim=-1)), dim=-1)
    # Insert a duplicate box waypoint at the end of the pickup dwell. This
    # represents waiting in time without asking ms18 for an unseen zero-speed gait.
    before = base[..., :17]
    after = base[..., 17:] + dwell[..., None]
    times = torch.cat((before, base[..., 16:17] + dwell[..., None], after), dim=-1)
    points = torch.cat((path[..., :17, :], path[..., 16:17, :], path[..., 17:, :]), dim=-2)
    return points, times


def _sample_at_time(points: torch.Tensor, times: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    flat_t = times.reshape(-1, times.shape[-1]).contiguous()
    flat_q = query.reshape(-1, query.shape[-1]).contiguous()
    idx = torch.searchsorted(flat_t, flat_q, right=True).sub(1).clamp(0, points.shape[-2] - 2)
    row = torch.arange(flat_t.shape[0], device=points.device)[:, None]
    t0 = flat_t.gather(1, idx)
    t1 = flat_t.gather(1, idx + 1)
    frac = ((flat_q - t0) / (t1 - t0).clamp(min=1e-5)).clamp(0.0, 1.0)
    feature_dim = points.shape[-1]
    flat_p = points.reshape(-1, points.shape[-2], feature_dim)
    out = flat_p[row, idx] + frac[..., None] * (flat_p[row, idx + 1] - flat_p[row, idx])
    return out.reshape(*query.shape, feature_dim)


def sample_timed_trajectory(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    state: CoordinatorState,
    query_seconds: torch.Tensor,
    *,
    measured_executor_timing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Sample joint position/speed at fixed future seconds.

    ``query_seconds`` is one-dimensional and shared by the batch.  The helper
    is used by temporal consistency training; unlike ``timed_rollout`` it does
    not normalize time by each candidate's makespan.
    """
    if query_seconds.ndim != 1 or query_seconds.numel() == 0:
        raise ValueError("query_seconds must be a non-empty 1-D tensor")
    if (query_seconds < 0.0).any():
        raise ValueError("query_seconds must be non-negative")
    points, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    query = query_seconds.to(device=path.device, dtype=path.dtype).reshape(
        1, 1, 1, -1
    ).expand(path.shape[0], path.shape[1], AGENTS, -1)
    position = _sample_at_time(points, arrival, query)
    expanded_speed = torch.cat(
        (speed[..., :17], speed[..., 16:17], speed[..., 17:]), dim=-1
    )
    sampled_speed = _sample_at_time(
        expanded_speed.unsqueeze(-1), arrival, query
    ).squeeze(-1)
    valid = query <= arrival[..., -1, None]
    return {
        "position": position,
        "speed": sampled_speed,
        "valid": valid,
        "duration": arrival[..., -1],
        "query_time": query,
    }


def trajectory_duration(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    state: CoordinatorState,
    *,
    measured_executor_timing: bool = False,
) -> torch.Tensor:
    """Return predicted completion seconds for every candidate and agent."""
    _, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    return arrival[..., -1]


def timed_rollout(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    state: CoordinatorState,
    samples: int = 96,
    measured_executor_timing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Time-align humans and carried boxes for all BxK candidates."""
    points, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    duration = arrival[..., -1]
    tmax = duration.amax(dim=-1)
    unit = torch.linspace(0.0, 1.0, samples, device=path.device, dtype=path.dtype)
    query = tmax[..., None, None] * unit.reshape(1, 1, 1, -1)
    query = query.expand(-1, -1, AGENTS, -1)
    root = _sample_at_time(points, arrival, query)

    pickup_time = arrival[..., 17]
    carrying = query >= pickup_time[..., None]
    initial_box = state.box_xyz[..., :2][:, None, :, None, :]
    box = torch.where(carrying[..., None], root, initial_box)
    return {"root": root, "box": box, "duration": duration, "query_time": query}


def pointwise_proximity_risk(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    state: CoordinatorState,
    *,
    human_radius: float = 0.35,
    proximity_margin: float = 0.0,
    proximity_approach_beta: float = 1.0,
    measured_executor_timing: bool = False,
) -> torch.Tensor:
    """Return continuous time-aligned risk at every agent path point.

    Each agent's path points are queried against the other agent's trajectory
    at the same predicted time. The radius expands after pickup using the
    actual box size. No crossing label, scenario id, or target speed is used.
    """
    if proximity_margin < 0.0 or proximity_approach_beta < 0.0:
        raise ValueError("proximity parameters must be non-negative")
    points, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    # Remove the inserted post-dwell duplicate to recover one time per input
    # path point: [0..16] before pickup, [17..32] after pickup.
    point_time = torch.cat((arrival[..., :17], arrival[..., 18:]), dim=-1)
    other_position = _sample_at_time(
        points.flip(dims=(2,)), arrival.flip(dims=(2,)), point_time
    )
    distance = (path - other_position).norm(dim=-1)

    pickup_time = arrival[..., 17]
    own_carrying = (
        (state.held[:, None, :, None] >= 0.5)
        | (point_time >= pickup_time[..., None])
    )
    other_pickup = pickup_time.flip(dims=(2,))
    other_carrying = (
        (state.held.flip(dims=(1,))[:, None, :, None] >= 0.5)
        | (point_time >= other_pickup[..., None])
    )
    box_radius = 0.5 * state.box_size_xy.norm(dim=-1)
    own_radius = (
        float(human_radius) + float(proximity_margin)
        + own_carrying.to(path.dtype) * box_radius[:, None, :, None]
    )
    other_radius = (
        float(human_radius) + float(proximity_margin)
        + other_carrying.to(path.dtype)
        * box_radius.flip(dims=(1,))[:, None, :, None]
    )
    radius_sum = (own_radius + other_radius).clamp(min=1e-5)
    normalized_distance = distance / radius_sum
    base = torch.exp(-0.5 * normalized_distance.square())
    closing = F.relu(distance[..., :-1] - distance[..., 1:]) / radius_sum[
        ..., :-1
    ]
    closing = torch.cat((closing, torch.zeros_like(closing[..., :1])), dim=-1)
    return base * (1.0 + float(proximity_approach_beta) * closing)


def crossing_arrival_metrics(
    path: torch.Tensor,
    speed: torch.Tensor,
    dwell: torch.Tensor,
    fixed_yield_agent1: bool = False,
    *,
    state: CoordinatorState | None = None,
    measured_executor_timing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Measure the time separation at the closest geometric route crossing.

    A pure time-aligned distance loss is diluted by all safe samples before and
    after a brief crossing.  For the toy Cross task, locate the closest pair of
    route points first and then compare when the two agents reach those points.
    Whichever agent is already later remains the yielding agent; exact ties are
    broken toward agent 1.  The discrete choice is detached, while arrival time
    remains differentiable with respect to path length and speed.
    """
    points, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    p0, p1 = points[:, :, 0], points[:, :, 1]
    batch, candidates, vertices, _ = p0.shape
    pair_distance = torch.cdist(
        p0.reshape(batch * candidates, vertices, 2),
        p1.reshape(batch * candidates, vertices, 2),
    ).reshape(batch, candidates, vertices, vertices)
    flat_index = pair_distance.flatten(start_dim=-2).argmin(dim=-1)
    i0 = torch.div(flat_index, vertices, rounding_mode="floor")
    i1 = flat_index % vertices
    t0 = arrival[:, :, 0].gather(-1, i0[..., None]).squeeze(-1)
    t1 = arrival[:, :, 1].gather(-1, i1[..., None]).squeeze(-1)
    yield_agent1 = (
        torch.ones_like(t1, dtype=torch.bool)
        if fixed_yield_agent1 else (t1.detach() >= t0.detach())
    )
    ordered_gap = torch.where(yield_agent1, t1 - t0, t0 - t1)
    return {
        "crossing_distance": pair_distance.flatten(start_dim=-2).amin(dim=-1),
        "crossing_time_gap": ordered_gap,
        "crossing_arrival_delta": t1 - t0,
        "crossing_arrival_t0": t0,
        "crossing_arrival_t1": t1,
        "crossing_index0": i0,
        "crossing_index1": i1,
        "yield_agent1": yield_agent1,
    }


def candidate_costs(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    human_clearance: float = 1.0,
    human_radius: float = 0.35,
    box_margin: float = 0.15,
    time_uncertainty: float = 0.0,
    measured_executor_timing: bool = False,
    proximity_approach_beta: float = 1.0,
    proximity_margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if time_uncertainty < 0.0:
        raise ValueError("time_uncertainty must be non-negative")
    if proximity_approach_beta < 0.0:
        raise ValueError("proximity_approach_beta must be non-negative")
    if proximity_margin < 0.0:
        raise ValueError("proximity_margin must be non-negative")
    path = output["path_world"]
    candidates = path.shape[1]
    speed = output["speed"]
    dwell = output["pickup_dwell"]
    rollout = timed_rollout(
        path, speed, dwell, state,
        measured_executor_timing=measured_executor_timing,
    )
    root, box = rollout["root"], rollout["box"]
    hh = (root[:, :, 0] - root[:, :, 1]).norm(dim=-1)
    bb = (box[:, :, 0] - box[:, :, 1]).norm(dim=-1)
    hb01 = (root[:, :, 0] - box[:, :, 1]).norm(dim=-1)
    hb10 = (root[:, :, 1] - box[:, :, 0]).norm(dim=-1)
    box_radius = 0.5 * state.box_size_xy.norm(dim=-1)
    bb_limit = box_radius.sum(dim=-1)[:, None, None] + box_margin
    hb01_limit = human_radius + box_radius[:, 1, None, None]
    hb10_limit = human_radius + box_radius[:, 0, None, None]

    hh_violation = F.relu(human_clearance - hh).square().mean(dim=-1)
    bb_violation = F.relu(bb_limit - bb).square().mean(dim=-1)
    hb_violation = 0.5 * (
        F.relu(hb01_limit - hb01).square().mean(dim=-1)
        + F.relu(hb10_limit - hb10).square().mean(dim=-1)
    )
    collision_steps = (
        F.relu(human_clearance - hh).square()
        + F.relu(bb_limit - bb).square()
        + 0.5 * (
            F.relu(hb01_limit - hb01).square()
            + F.relu(hb10_limit - hb10).square()
        )
    )
    future_collision_steps = collision_steps

    # General, scenario-agnostic proximity risk.  Each agent is represented by
    # its root plus a larger envelope after pickup.  Unlike the legacy hinge,
    # this provides a smooth gradient before physical overlap.  Closing motion
    # strengthens the signal, while separating motion receives no multiplier.
    pickup_time = rollout["query_time"].new_zeros(
        rollout["query_time"].shape[:3]
    )
    _, arrival = _arrival_times(
        path, speed, dwell, state=state,
        measured_executor_timing=measured_executor_timing,
    )
    pickup_time.copy_(arrival[..., 17])
    carrying = (
        (state.held[:, None, :, None] >= 0.5)
        | (rollout["query_time"] >= pickup_time[..., None])
    )
    dynamic_radius = human_radius + float(proximity_margin) + carrying.to(path.dtype) * box_radius[
        :, None, :, None
    ]
    radius_sum = dynamic_radius[:, :, 0] + dynamic_radius[:, :, 1]
    normalized_distance = hh / radius_sum.clamp(min=1e-5)
    proximity_base = torch.exp(-0.5 * normalized_distance.square())
    closing = F.relu(hh[..., :-1] - hh[..., 1:]) / radius_sum[..., :-1].clamp(
        min=1e-5
    )
    closing = torch.cat((closing, torch.zeros_like(closing[..., :1])), dim=-1)
    proximity_collision_steps = proximity_base * (
        1.0 + float(proximity_approach_beta) * closing
    )
    if time_uncertainty > 0.0:
        # The frozen executor does not arrive at a waypoint at exactly the
        # commanded kinematic time, especially around pickup. Penalize close
        # encounters whose two timestamps differ by at most the uncertainty
        # band. This remains fully trajectory-based: no yielder or target
        # speed is supplied.
        root0, root1 = root[:, :, 0], root[:, :, 1]
        box0, box1 = box[:, :, 0], box[:, :, 1]
        time0 = rollout["query_time"][:, :, 0]
        time1 = rollout["query_time"][:, :, 1]
        time_mask = (
            (time0[..., :, None] - time1[..., None, :]).abs()
            <= float(time_uncertainty)
        ).to(path.dtype)
        hh_pair = torch.cdist(root0, root1)
        bb_pair = torch.cdist(box0, box1)
        hb01_pair = torch.cdist(root0, box1)
        # Axes stay (agent0 time, agent1 time).
        hb10_pair = torch.cdist(box0, root1)
        robust = (
            F.relu(human_clearance - hh_pair).square()
            + F.relu(bb_limit[..., None] - bb_pair).square()
            + 0.5 * (
                F.relu(hb01_limit[..., None] - hb01_pair).square()
                + F.relu(hb10_limit[..., None] - hb10_pair).square()
            )
        )
        future_collision_steps = (robust * time_mask).flatten(start_dim=-2)
    peak_collision = (
        F.relu(human_clearance - hh).square().amax(dim=-1)
        + F.relu(bb_limit - bb).square().amax(dim=-1)
        + 0.5 * (
            F.relu(hb01_limit - hb01).square().amax(dim=-1)
            + F.relu(hb10_limit - hb10).square().amax(dim=-1)
        )
    )
    min_hh = hh.amin(dim=-1)
    min_bb_margin = (bb - bb_limit).amin(dim=-1)
    min_hb_margin = torch.minimum(
        (hb01 - hb01_limit).amin(dim=-1),
        (hb10 - hb10_limit).amin(dim=-1),
    )

    segment = path[..., 1:, :] - path[..., :-1, :]
    agent_length = segment.norm(dim=-1).sum(dim=-1)
    length = agent_length.sum(dim=-1)
    makespan = rollout["duration"].amax(dim=-1)
    speed_roughness = (speed[..., 1:] - speed[..., :-1]).square().mean(dim=-1).mean(dim=-1)
    v0, v1 = segment[..., :-1, :], segment[..., 1:, :]
    cos = (v0 * v1).sum(dim=-1) / (v0.norm(dim=-1) * v1.norm(dim=-1)).clamp(min=1e-7)
    turn = torch.acos(cos.clamp(-1.0, 1.0))
    # The box anchor can be a deliberate corner between approach and carry.
    turn[..., 14:17] = 0.0
    held_turn = state.held[:, None, :, None] >= 0.5
    turn[..., :16] = torch.where(
        held_turn.expand(-1, candidates, -1, 16),
        torch.zeros_like(turn[..., :16]),
        turn[..., :16],
    )
    max_turn = torch.rad2deg(turn.flatten(start_dim=2).amax(dim=-1))
    constraint_cos = cos.clone()
    constraint_cos[..., 14:17] = 1.0
    constraint_cos[..., :16] = torch.where(
        held_turn.expand(-1, candidates, -1, 16),
        torch.ones_like(constraint_cos[..., :16]),
        constraint_cos[..., :16],
    )
    min_turn_cos = constraint_cos.flatten(start_dim=2).amin(dim=-1)

    finite = torch.isfinite(path).flatten(start_dim=2).all(dim=-1)
    finite &= torch.isfinite(speed).flatten(start_dim=2).all(dim=-1)
    # The ms18 bridge owns one 320-cell (31.8 m usable) buffer per agent.
    # A short path from one agent must not hide overflow from the other.
    buffer_ok = (agent_length < 31.8).all(dim=-1)
    curvature_ok = max_turn <= 46.0
    valid = finite & buffer_ok & curvature_ok
    safe = (min_hh >= human_clearance) & (min_bb_margin >= 0.0) & (min_hb_margin >= 0.0)
    collision = hh_violation + bb_violation + hb_violation
    # Candidate choice is based on the decoded future itself.  Do not inject the
    # critic estimate here: PPO trains its batch mean as a state value, so its
    # candidate-wise offsets are not calibrated ranking scores.
    raw_cost = 100.0 * collision + makespan + 0.05 * length + 0.25 * speed_roughness
    # The selector needs a hard ban, while learning needs a finite slope back
    # toward feasibility.  Sharing the 1e9 sentinel made all-invalid samples
    # dominate the reported auxiliary loss without providing any gradient.
    # Use cosine space for the training penalty.  Differentiating acos at a
    # perfectly straight segment (cos=1) produces an infinite derivative.
    cos_limit = math.cos(math.radians(46.0))
    curvature_penalty = 20.0 * F.relu(cos_limit - min_turn_cos).square()
    buffer_penalty = 10.0 * F.relu(agent_length - 31.8).square().sum(dim=-1)
    train_cost = torch.nan_to_num(
        raw_cost + curvature_penalty + buffer_penalty,
        nan=1e6,
        posinf=1e6,
        neginf=1e6,
    )
    cost = torch.where(valid, raw_cost, torch.full_like(raw_cost, 1e9))
    # Safety is lexicographic: any valid safe future beats any unsafe future.
    rank_cost = cost + (~safe).to(cost.dtype) * 1e6
    return {
        "cost": cost,
        "train_cost": train_cost,
        "rank_cost": rank_cost,
        "valid": valid,
        "safe": safe,
        "collision": collision,
        "collision_steps": collision_steps,
        "future_collision_steps": future_collision_steps,
        "proximity_collision_steps": proximity_collision_steps,
        "proximity_collision": proximity_collision_steps.mean(dim=-1),
        "proximity_closing": closing.mean(dim=-1),
        "peak_collision": peak_collision,
        "min_hh": min_hh,
        "min_bb_margin": min_bb_margin,
        "min_hb_margin": min_hb_margin,
        "makespan": makespan,
        "length": length,
        "speed_roughness": speed_roughness,
        "max_turn_deg": max_turn,
        "measured_executor_timing": path.new_tensor(
            float(measured_executor_timing)
        ),
    }


def _gather_candidate(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    row = torch.arange(value.shape[0], device=value.device)
    return value[row, index]


def select_candidate(output: Dict[str, torch.Tensor], state: CoordinatorState) -> CandidateSelection:
    if output["path_world"].shape[1] <= 0:
        raise ValueError("candidate axis must be non-empty")
    diagnostics = candidate_costs(output, state)
    index = diagnostics["rank_cost"].argmin(dim=1)
    selected_valid = _gather_candidate(diagnostics["valid"], index)
    selected_safe = _gather_candidate(diagnostics["safe"], index)
    return CandidateSelection(
        index=index,
        path_world=_gather_candidate(output["path_world"], index),
        speed=_gather_candidate(output["speed"], index),
        pickup_dwell=_gather_candidate(output["pickup_dwell"], index),
        valid=selected_valid,
        safe=selected_safe,
        diagnostics=diagnostics,
    )
