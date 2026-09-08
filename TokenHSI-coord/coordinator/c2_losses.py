"""Minimal C2 objective for anticipatory path/speed coordination."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .planner import (
    candidate_costs,
    crossing_arrival_metrics,
    pointwise_proximity_risk,
    sample_timed_trajectory,
    trajectory_duration,
)
from .schema import MAX_SPEED, MIN_SPEED, CoordinatorState


FUTURE_COLLISION_COEF = 50.0
SPEED_SMOOTH_COEF = 0.5
UNNECESSARY_SLOW_COEF = 0.1
ARRIVAL_GAP_COEF = 2.0
PRIORITY_SPEED_COEF = 5.0
PROFILE_EFFICIENCY_COEF = 1.0
CONSISTENCY_START_S = 0.5
CONSISTENCY_STEP_S = 0.25
CONSISTENCY_SAMPLES = 32
CONSISTENCY_SPEED_WEIGHT = 0.25


def _consistency_query(reference: torch.Tensor) -> torch.Tensor:
    return CONSISTENCY_START_S + CONSISTENCY_STEP_S * torch.arange(
        CONSISTENCY_SAMPLES, device=reference.device, dtype=reference.dtype
    )


def _straight_paths(state: CoordinatorState, candidates: int) -> torch.Tensor:
    """Return hard root->box->goal paths as [B,K,A,33,2]."""
    unit = torch.linspace(
        0.0, 1.0, 17, device=state.device, dtype=state.root_xy.dtype
    )
    root, box, goal = state.root_xy, state.box_xyz[..., :2], state.goal_xy
    approach = root[:, :, None] + unit[None, None, :, None] * (
        box - root
    )[:, :, None]
    carry = box[:, :, None] + unit[None, None, :, None] * (
        goal - box
    )[:, :, None]
    path = torch.cat((approach, carry[:, :, 1:]), dim=-2)[:, None]
    return path.expand(-1, candidates, -1, -1, -1)


@torch.no_grad()
def build_trajectory_consistency_target(
    previous_output: Dict[str, torch.Tensor],
    previous_state: CoordinatorState,
    current_state: CoordinatorState,
    elapsed_seconds: float,
    active_env: torch.Tensor,
    *,
    measured_executor_timing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Advance the previous plan in time and detach it as a training target."""
    if elapsed_seconds < 0.0:
        raise ValueError("elapsed_seconds must be non-negative")
    if active_env.shape != (current_state.batch_size,):
        raise ValueError("active_env must be [B]")
    query = _consistency_query(previous_output["speed"]) + elapsed_seconds
    sampled = sample_timed_trajectory(
        previous_output["path_world"], previous_output["speed"],
        previous_output["pickup_dwell"], previous_state, query,
        measured_executor_timing=measured_executor_timing,
    )
    same_phase = current_state.phase == previous_state.phase
    valid = (
        sampled["valid"]
        & active_env[:, None, None, None].bool()
        & same_phase[:, None, :, None]
    )
    return {
        "position": sampled["position"].detach(),
        "speed": sampled["speed"].detach(),
        "valid": valid.detach(),
    }


def trajectory_consistency_loss(
    output: Dict[str, torch.Tensor],
    state: CoordinatorState,
    target: Optional[Dict[str, torch.Tensor]],
    *,
    measured_executor_timing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compare a new plan with the time-shifted remaining previous plan."""
    zero = output["speed"].new_zeros(())
    if target is None:
        return {
            "total": zero,
            "position": zero,
            "speed": zero,
            "valid_fraction": zero,
        }
    sampled = sample_timed_trajectory(
        output["path_world"], output["speed"], output["pickup_dwell"],
        state, _consistency_query(output["speed"]),
        measured_executor_timing=measured_executor_timing,
    )
    expected_shapes = {
        "position": sampled["position"].shape,
        "speed": sampled["speed"].shape,
        "valid": sampled["valid"].shape,
    }
    for key, shape in expected_shapes.items():
        if key not in target or target[key].shape != shape:
            raise ValueError(
                f"consistency target {key}: expected {tuple(shape)}, "
                f"got {None if key not in target else tuple(target[key].shape)}"
            )
    mask = sampled["valid"] & target["valid"].bool()
    weight = mask.to(output["speed"].dtype)
    denominator = weight.sum().clamp(min=1.0)
    position_error = F.smooth_l1_loss(
        sampled["position"], target["position"], reduction="none",
        beta=0.25,
    ).mean(dim=-1)
    position = (weight * position_error).sum() / denominator
    speed = (
        weight * (sampled["speed"] - target["speed"]).square()
    ).sum() / denominator
    total = position + CONSISTENCY_SPEED_WEIGHT * speed
    return {
        "total": total,
        "position": position,
        "speed": speed,
        "valid_fraction": weight.mean(),
    }


def compute_c2_auxiliary_loss(
    output: Dict[str, torch.Tensor], state: CoordinatorState, *,
    peak_collision: bool = False, human_clearance: float = 1.0,
    arrival_gap: bool = False, arrival_time_margin: float = 1.0,
    arrival_gap_coef: float = ARRIVAL_GAP_COEF,
    fixed_yield_agent1: bool = False,
    bidirectional_target_gap: bool = False,
    state_ordered_target_gap: bool = False,
    fixed_yield_speed: float = 0.0,
    path_collision_only: bool = False,
    collision_focus_steps: int = 0,
    collision_time_uncertainty: float = 0.0,
    proximity_collision: bool = False,
    proximity_approach_beta: float = 1.0,
    proximity_margin: float = 0.0,
    future_collision_coef: float = FUTURE_COLLISION_COEF,
    speed_efficiency_coef: float = 0.0,
    path_residual_coef: float = 0.0,
    path_smooth_coef: float = 0.0,
    measured_executor_timing: bool = False,
    consistency_target: Optional[Dict[str, torch.Tensor]] = None,
    consistency_coef: float = 0.0,
    speed_smooth_coef: float = SPEED_SMOOTH_COEF,
    unnecessary_slow_coef: float = UNNECESSARY_SLOW_COEF,
    extra_delay_coef: float = 0.0,
    detour_delay_coef: float = 0.0,
    dual_delay_coef: float = 0.0,
    priority_agent: Optional[torch.Tensor] = None,
    explicit_gap_coef: float = 0.0,
    explicit_anchor_coef: float = 0.0,
    explicit_post_coef: float = 0.0,
    initial_plan_mask: Optional[torch.Tensor] = None,
    initial_plan_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Penalize predicted collision, jerky speed, and slowing in safe scenes.

    The full-speed counterfactual uses the predicted path but replaces its speed
    with 1.5 m/s.  Slowdown is free only when that same path would conflict at
    full speed; otherwise the policy is gently pulled back to nominal speed.
    """
    if output["path_world"].shape[1] != 1:
        raise ValueError("C2 requires exactly one joint trajectory (K=1)")
    if fixed_yield_speed != 0.0 and not MIN_SPEED <= fixed_yield_speed <= MAX_SPEED:
        raise ValueError(
            f"fixed_yield_speed must be 0(disabled) or [{MIN_SPEED}, {MAX_SPEED}]"
        )
    if collision_focus_steps < 0:
        raise ValueError("collision_focus_steps must be non-negative")
    if collision_time_uncertainty < 0.0:
        raise ValueError("collision_time_uncertainty must be non-negative")
    if proximity_approach_beta < 0.0:
        raise ValueError("proximity_approach_beta must be non-negative")
    if proximity_margin < 0.0:
        raise ValueError("proximity_margin must be non-negative")
    if future_collision_coef < 0.0:
        raise ValueError("future_collision_coef must be non-negative")
    if speed_efficiency_coef < 0.0:
        raise ValueError("speed_efficiency_coef must be non-negative")
    if consistency_coef < 0.0:
        raise ValueError("consistency_coef must be non-negative")
    if speed_smooth_coef < 0.0 or unnecessary_slow_coef < 0.0:
        raise ValueError("speed regularizer coefficients must be non-negative")
    if extra_delay_coef < 0.0 or detour_delay_coef < 0.0:
        raise ValueError("delay coefficients must be non-negative")
    if dual_delay_coef < 0.0:
        raise ValueError("dual_delay_coef must be non-negative")
    if min(explicit_gap_coef, explicit_anchor_coef, explicit_post_coef) < 0.0:
        raise ValueError("explicit speed coefficients must be non-negative")
    if initial_plan_weight < 1.0:
        raise ValueError("initial_plan_weight must be at least one")

    metrics = candidate_costs(
        output, state, human_clearance=human_clearance,
        time_uncertainty=collision_time_uncertainty,
        measured_executor_timing=measured_executor_timing,
        proximity_approach_beta=proximity_approach_beta,
        proximity_margin=proximity_margin,
    )
    collision_steps_key = (
        "proximity_collision_steps"
        if proximity_collision else "future_collision_steps"
    )
    if collision_focus_steps:
        if collision_focus_steps > metrics[collision_steps_key].shape[-1]:
            raise ValueError("collision_focus_steps exceeds timed rollout samples")
        future_collision = metrics[collision_steps_key].topk(
            collision_focus_steps, dim=-1
        ).values.mean()
    else:
        future_collision = (
            metrics["proximity_collision"].mean()
            if proximity_collision
            else metrics["peak_collision" if peak_collision else "collision"].mean()
        )
    raw_proximity_collision = metrics["proximity_collision"].mean()
    speed = output["speed"]
    speed_smooth = (speed[..., 1:] - speed[..., :-1]).square().mean()
    if "waypoint_residual" in output:
        waypoint_residual = output["waypoint_residual"]
        path_residual = waypoint_residual.square().mean()
        # Do not smooth across the hard box anchor: the approach and carry
        # legs are two independent curves which merely share P16=box.
        zero = torch.zeros_like(waypoint_residual[..., :1, :])
        approach = torch.cat((zero, waypoint_residual[..., :15, :], zero), dim=-2)
        carry = torch.cat((zero, waypoint_residual[..., 15:, :], zero), dim=-2)
        approach_d2 = approach[..., 2:, :] - 2.0 * approach[..., 1:-1, :] + approach[..., :-2, :]
        carry_d2 = carry[..., 2:, :] - 2.0 * carry[..., 1:-1, :] + carry[..., :-2, :]
        # A raw metre-valued second difference is too permissive on a short
        # pickup leg: the same 1 cm kink is harmless on a 50 cm segment but
        # can exceed the bridge's 46-degree limit on a 5 cm segment. Normalize
        # by each leg's nominal point spacing to penalize angular roughness.
        approach_step = (
            (state.box_xyz[..., :2] - state.root_xy).norm(dim=-1) / 16.0
        ).clamp(min=0.05)[:, None, :, None, None]
        carry_step = (
            (state.goal_xy - state.box_xyz[..., :2]).norm(dim=-1) / 16.0
        ).clamp(min=0.05)[:, None, :, None, None]
        path_smooth = torch.cat(
            (approach_d2 / approach_step, carry_d2 / carry_step), dim=-2
        ).square().mean()
    else:
        # Bezier C2 fixes pickup straight and learns only carry controls.
        path_residual = output["control_residual"][..., 2:, :].square().mean()
        path_smooth = speed.new_zeros(())

    nominal_output = dict(output)
    nominal_output["speed"] = torch.full_like(speed, MAX_SPEED)
    nominal_point_risk = speed.new_zeros(())
    slowdown_request_mean = speed.new_zeros(())
    with torch.no_grad():
        nominal_metrics = candidate_costs(
            nominal_output, state, human_clearance=human_clearance,
            measured_executor_timing=measured_executor_timing,
        )
        nominal_conflict = nominal_metrics["collision"] > 1e-6
        safe_weight = (~nominal_conflict).to(speed.dtype)
    speed_deficit = (MAX_SPEED - speed).square().mean(dim=(-1, -2))
    speed_efficiency = speed_deficit.mean()
    if "slowdown_request" in output:
        # Localize the learned cause, not the physically derived braking tail.
        # Collision loss may activate q at risky points; this term removes q
        # everywhere the full-speed counterfactual is already safe.
        with torch.no_grad():
            point_risk = pointwise_proximity_risk(
                output["path_world"], nominal_output["speed"],
                output["pickup_dwell"], state,
                proximity_margin=proximity_margin,
                proximity_approach_beta=proximity_approach_beta,
                measured_executor_timing=measured_executor_timing,
            ).clamp(0.0, 1.0)
        request = output["slowdown_request"]
        if priority_agent is None:
            request_weight = torch.ones_like(request)
        else:
            agent = torch.arange(2, device=speed.device)
            request_weight = (
                agent[None] != priority_agent[:, None]
            ).to(speed.dtype)[:, None, :, None].expand_as(request)
        denominator = request_weight.sum().clamp(min=1.0)
        unnecessary_slow = (
            request_weight * request * (1.0 - point_risk)
        ).sum() / denominator
        nominal_point_risk = (
            request_weight * point_risk
        ).sum() / denominator
        slowdown_request_mean = (
            request_weight * request
        ).sum() / denominator
    else:
        unnecessary_slow = (safe_weight * speed_deficit).mean()
    duration = trajectory_duration(
        output["path_world"], speed, output["pickup_dwell"], state,
        measured_executor_timing=measured_executor_timing,
    )
    nominal_duration = trajectory_duration(
        output["path_world"], nominal_output["speed"],
        output["pickup_dwell"], state,
        measured_executor_timing=measured_executor_timing,
    )
    delay = (duration - nominal_duration).clamp(min=0.0)
    # No yielder label is supplied. Penalizing the product makes it cheap to
    # put necessary delay on either agent, but expensive to distribute delay
    # across both agents in the same joint plan. This provides a learned,
    # state-dependent symmetry break without a fixed-priority projection.
    candidate_agent_delay = delay.mean(dim=1)
    dual_delay = (
        candidate_agent_delay[:, 0] * candidate_agent_delay[:, 1]
    ).mean()
    if priority_agent is None:
        extra_delay = delay.mean()
    else:
        if priority_agent.shape != (state.batch_size,):
            raise ValueError("priority_agent must be [B]")
        agent = torch.arange(2, device=speed.device)
        yield_mask = (agent[None] != priority_agent[:, None]).to(speed.dtype)
        yield_mask = yield_mask[:, None].expand_as(delay)
        extra_delay = (yield_mask * delay).sum() / yield_mask.sum().clamp(min=1.0)

    detour_delay = speed.new_zeros(())
    if detour_delay_coef > 0.0:
        straight = _straight_paths(state, speed.shape[1])
        straight_duration = trajectory_duration(
            straight, nominal_output["speed"], output["pickup_dwell"], state,
            measured_executor_timing=measured_executor_timing,
        ).detach()
        detour = (nominal_duration - straight_duration).clamp(min=0.0)
        if priority_agent is None:
            detour_delay = detour.mean()
        else:
            agent = torch.arange(2, device=speed.device)
            yield_mask = (agent[None] != priority_agent[:, None]).to(speed.dtype)
            yield_mask = yield_mask[:, None].expand_as(detour)
            detour_delay = (
                yield_mask * detour
            ).sum() / yield_mask.sum().clamp(min=1.0)

    crossing = crossing_arrival_metrics(
        output["path_world"], speed, output["pickup_dwell"],
        fixed_yield_agent1=fixed_yield_agent1,
        state=state,
        measured_executor_timing=measured_executor_timing,
    )
    crossing_conflict = (
        crossing["crossing_distance"].detach() < human_clearance
    ).to(speed.dtype)
    gap_for_loss = crossing["crossing_time_gap"]
    if fixed_yield_agent1:
        # Do not reverse an already safe natural ordering caused by different
        # pickup phases.  Only near-synchronous crossings need the fixed a1
        # tie-break; either ordering is acceptable once the margin is met.
        delta = crossing["crossing_arrival_delta"]
        already_separated = delta.detach().abs() >= arrival_time_margin
        gap_for_loss = torch.where(already_separated, delta.abs(), delta)
    arrival_gap_loss = (
        crossing_conflict
        * torch.relu(arrival_time_margin - gap_for_loss).square()
    ).mean()
    crossing_time_gap = (
        crossing_conflict * gap_for_loss.detach()
    ).sum() / crossing_conflict.sum().clamp(min=1.0)
    priority_speed = (MAX_SPEED - speed[:, :, 0]).square().mean()

    nominal_crossing = crossing_arrival_metrics(
        output["path_world"], nominal_output["speed"], output["pickup_dwell"],
        state=state,
        measured_executor_timing=measured_executor_timing,
    )
    nominal_delta = nominal_crossing["crossing_arrival_delta"].detach()
    target_gap_needed = (
        (nominal_crossing["crossing_distance"].detach() < human_clearance)
        & (nominal_delta.abs() < arrival_time_margin)
    ).to(speed.dtype)
    actual_delta = crossing["crossing_arrival_delta"]
    deficit_per_agent = (MAX_SPEED - speed).square().mean(dim=-1)
    a1_yields = (
        (actual_delta - arrival_time_margin).square()
        + PRIORITY_SPEED_COEF * deficit_per_agent[:, :, 0]
    )
    a0_yields = (
        (actual_delta + arrival_time_margin).square()
        + PRIORITY_SPEED_COEF * deficit_per_agent[:, :, 1]
    )
    # Choosing from the policy's current predictions creates a positive
    # feedback loop: a tiny initial preference for one slot makes that same
    # slot the cheaper yielder forever.  The state-ordered mode instead uses
    # the full-speed counterfactual.  Whichever agent is naturally later at
    # the crossing yields, so both slots are trainable across scenes and the
    # requested delay is minimal.
    if state_ordered_target_gap:
        choose_a1 = nominal_delta >= 0.0
    else:
        choose_a1 = a1_yields.detach() <= a0_yields.detach()
    chosen_yield_loss = torch.where(choose_a1, a1_yields, a0_yields)
    bidirectional_gap_loss = (
        target_gap_needed * chosen_yield_loss
    ).mean()
    profile_efficiency = deficit_per_agent.mean()
    chosen_gap = torch.where(choose_a1, actual_delta, -actual_delta)
    target_gap_actual = (
        target_gap_needed * chosen_gap.detach()
    ).sum() / target_gap_needed.sum().clamp(min=1.0)
    yield_agent1_fraction = (
        target_gap_needed * choose_a1.to(speed.dtype)
    ).sum() / target_gap_needed.sum().clamp(min=1.0)

    # A scalar arrival-time objective is underdetermined: a receding-horizon
    # policy can place a narrow slowdown in the middle of every newly planned
    # route and postpone it forever.  Build the simplest exact supervision for
    # the state-ordered mode instead.  The naturally later agent travels at one
    # constant reduced speed from *now* through the crossing, then returns to
    # 1.5 m/s.  The executor's temporal rate limiter smooths the actual change.
    nominal_t = torch.stack(
        (nominal_crossing["crossing_arrival_t0"],
         nominal_crossing["crossing_arrival_t1"]),
        dim=-1,
    ).detach()
    expanded_index = torch.stack(
        (nominal_crossing["crossing_index0"],
         nominal_crossing["crossing_index1"]),
        dim=-1,
    ).detach()
    # _arrival_times inserts a duplicate box vertex at index 17.
    path_index = (
        expanded_index - (expanded_index >= 17).to(expanded_index.dtype)
    ).clamp(0, speed.shape[-1] - 1)
    dwell_before = output["pickup_dwell"] * (expanded_index >= 17).to(speed.dtype)
    travel_distance = (nominal_t - dwell_before).clamp(min=0.0) * MAX_SPEED
    priority_t = torch.where(
        choose_a1[..., None], nominal_t[..., 0:1], nominal_t[..., 1:2]
    )
    target_arrival = priority_t + arrival_time_margin
    target_speed_value = (
        travel_distance / (target_arrival - dwell_before).clamp(min=1e-3)
    ).clamp(min=0.375, max=MAX_SPEED)
    if fixed_yield_speed > 0.0:
        # Toy collision-first supervision: make the yield command unambiguous.
        # The previous distance-derived target was usually about 1.2 m/s, which
        # is too close to the executor's natural gait to create a reliable time
        # gap.  This changes only the target; the policy still predicts one
        # speed at every waypoint from state.
        target_speed_value = torch.full_like(
            target_speed_value, float(fixed_yield_speed)
        )
    yielder = torch.stack((~choose_a1, choose_a1), dim=-1)
    point = torch.arange(speed.shape[-1], device=speed.device)
    prefix = point.reshape(1, 1, 1, -1) <= path_index[..., None]
    prefix_mask = (
        target_gap_needed[..., None, None]
        * yielder[..., None].to(speed.dtype)
        * prefix.to(speed.dtype)
    )
    prefix_target = torch.where(
        prefix_mask.bool(), target_speed_value[..., None],
        torch.full_like(speed, MAX_SPEED),
    )
    prefix_speed_target_loss = (
        prefix_mask * (speed - prefix_target).square()
    ).sum() / prefix_mask.sum().clamp(min=1.0)
    restore_mask = 1.0 - prefix_mask
    restore_speed_loss = (
        restore_mask * (speed - MAX_SPEED).square()
    ).sum() / restore_mask.sum().clamp(min=1.0)
    target_yield_speed = (
        target_gap_needed[..., None]
        * yielder.to(speed.dtype)
        * target_speed_value
    ).sum() / (
        target_gap_needed[..., None] * yielder.to(speed.dtype)
    ).sum().clamp(min=1.0)

    # Optional behavior-shaping baseline.  No target speed or oracle path is
    # supplied: a full-speed straight counterfactual only identifies the
    # potential crossing.  The assigned yielder must arrive later, its single
    # slowdown support is attached to the box on the crossing-facing side,
    # and its command must return to nominal after clearing the crossing.
    explicit_gap = speed.new_zeros(())
    explicit_anchor = speed.new_zeros(())
    explicit_post = speed.new_zeros(())
    explicit_need_rate = speed.new_zeros(())
    explicit_assigned_gap = speed.new_zeros(())
    explicit_enabled = any(
        coefficient > 0.0 for coefficient in (
            explicit_gap_coef, explicit_anchor_coef, explicit_post_coef
        )
    )
    if explicit_enabled:
        if priority_agent is None:
            raise ValueError("explicit speed shaping requires priority_agent")
        if "slowdown_window" not in output:
            raise ValueError("explicit speed shaping requires slowdown_window output")
        if initial_plan_mask is not None and initial_plan_mask.shape != (
            state.batch_size,
        ):
            raise ValueError("initial_plan_mask must be [B]")

        straight = _straight_paths(state, speed.shape[1])
        straight_nominal = crossing_arrival_metrics(
            straight, torch.full_like(speed, MAX_SPEED), output["pickup_dwell"],
            state=state, measured_executor_timing=measured_executor_timing,
        )
        straight_actual = crossing_arrival_metrics(
            straight, speed, output["pickup_dwell"], state=state,
            measured_executor_timing=measured_executor_timing,
        )

        candidates = speed.shape[1]
        priority_index = priority_agent[:, None, None].expand(-1, candidates, 1)
        yield_index = (1 - priority_agent)[:, None, None].expand(
            -1, candidates, 1
        )

        def pick_agent(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            expanded = index
            while expanded.ndim < value.ndim:
                expanded = expanded.unsqueeze(-1)
            expanded = expanded.expand(
                value.shape[0], value.shape[1], 1, *value.shape[3:]
            )
            return value.gather(2, expanded).squeeze(2)

        nominal_arrival = torch.stack(
            (
                straight_nominal["crossing_arrival_t0"],
                straight_nominal["crossing_arrival_t1"],
            ), dim=2,
        )
        actual_arrival = torch.stack(
            (
                straight_actual["crossing_arrival_t0"],
                straight_actual["crossing_arrival_t1"],
            ), dim=2,
        )
        nominal_gap = (
            pick_agent(nominal_arrival, yield_index)
            - pick_agent(nominal_arrival, priority_index)
        ).detach()
        assigned_gap = (
            pick_agent(actual_arrival, yield_index)
            - pick_agent(actual_arrival, priority_index)
        )
        needs_yield = (
            (straight_nominal["crossing_distance"].detach() < human_clearance)
            & (nominal_gap < arrival_time_margin)
        ).to(speed.dtype)
        batch_weight = torch.ones(
            state.batch_size, device=speed.device, dtype=speed.dtype
        )
        if initial_plan_mask is not None:
            batch_weight = torch.where(
                initial_plan_mask.bool(),
                torch.full_like(batch_weight, float(initial_plan_weight)),
                batch_weight,
            )
        sample_weight = batch_weight[:, None] * needs_yield
        denominator = sample_weight.sum().clamp(min=1.0)
        explicit_gap = (
            sample_weight
            * torch.relu(arrival_time_margin - assigned_gap).square()
        ).sum() / denominator
        explicit_need_rate = needs_yield.mean()
        explicit_assigned_gap = (
            needs_yield * assigned_gap.detach()
        ).sum() / needs_yield.sum().clamp(min=1.0)

        straight_ds = (
            straight[..., 1:, :] - straight[..., :-1, :]
        ).norm(dim=-1)
        straight_arc = torch.cat(
            (torch.zeros_like(straight_ds[..., :1]), straight_ds.cumsum(dim=-1)),
            dim=-1,
        )
        expanded_cross_index = torch.stack(
            (
                straight_nominal["crossing_index0"],
                straight_nominal["crossing_index1"],
            ), dim=2,
        )
        cross_index = (
            expanded_cross_index
            - (expanded_cross_index >= 17).to(expanded_cross_index.dtype)
        ).clamp(0, speed.shape[-1] - 1)
        yield_cross_index = pick_agent(cross_index, yield_index)
        yield_arc = pick_agent(straight_arc, yield_index)
        cross_s = yield_arc.gather(
            -1, yield_cross_index[..., None]
        ).squeeze(-1).detach()
        box_s = yield_arc[..., 16].detach()

        window = pick_agent(output["slowdown_window"], yield_index)
        left = window[..., 0] - 0.5 * window[..., 1]
        right = window[..., 0] + 0.5 * window[..., 1]
        box_facing_edge = torch.where(cross_s >= box_s, left, right)
        explicit_anchor = (
            sample_weight * (box_facing_edge - box_s).square()
        ).sum() / denominator

        yield_speed = pick_agent(speed, yield_index)
        post_mask = (
            yield_arc > (cross_s + human_clearance)[..., None]
        ).to(speed.dtype)
        post_weight = sample_weight[..., None] * post_mask
        explicit_post = (
            post_weight * (MAX_SPEED - yield_speed).square()
        ).sum() / post_weight.sum().clamp(min=1.0)

    if state_ordered_target_gap:
        if path_collision_only:
            # Train spatial avoidance against a full-speed counterfactual so
            # this term cannot lower both agents' speed.  The explicit prefix
            # target below is solely responsible for temporal yielding.
            path_collision_output = dict(output)
            path_collision_output["speed"] = torch.full_like(speed, MAX_SPEED)
            path_metrics = candidate_costs(
                path_collision_output, state, human_clearance=human_clearance,
                time_uncertainty=collision_time_uncertainty,
                measured_executor_timing=measured_executor_timing,
            )
            future_collision = path_metrics[
                "peak_collision" if peak_collision else "collision"
            ].mean()
        total = (
            arrival_gap_coef * (prefix_speed_target_loss + restore_speed_loss)
            + speed_smooth_coef * speed_smooth
            # Bezier experiments deliberately isolated speed coordination.
            # Direct-waypoint C2 is meant to choose between slowing and
            # spatial avoidance, so its path head must see the same future
            # time-aligned collision signal.
            + (future_collision_coef * future_collision
               if "waypoint_residual" in output else 0.0)
        )
    elif bidirectional_target_gap:
        total = (
            arrival_gap_coef * bidirectional_gap_loss
            + PROFILE_EFFICIENCY_COEF * profile_efficiency
            + speed_smooth_coef * speed_smooth
        )
    else:
        total = (
            (arrival_gap_coef * arrival_gap_loss if arrival_gap
             else future_collision_coef * future_collision)
            + (PRIORITY_SPEED_COEF * priority_speed
               if arrival_gap and fixed_yield_agent1 else 0.0)
            + speed_smooth_coef * speed_smooth
            + unnecessary_slow_coef * unnecessary_slow
            + speed_efficiency_coef * speed_efficiency
        )
    total = (
        total
        + path_residual_coef * path_residual
        + path_smooth_coef * path_smooth
        + extra_delay_coef * extra_delay
        + detour_delay_coef * detour_delay
        + dual_delay_coef * dual_delay
        + explicit_gap_coef * explicit_gap
        + explicit_anchor_coef * explicit_anchor
        + explicit_post_coef * explicit_post
    )
    consistency = trajectory_consistency_loss(
        output, state, consistency_target,
        measured_executor_timing=measured_executor_timing,
    )
    total = total + consistency_coef * consistency["total"]
    return {
        "total": total,
        "future_collision": future_collision,
        "speed_smooth": speed_smooth,
        "speed_smooth_coef": speed.new_tensor(float(speed_smooth_coef)),
        "unnecessary_slow": unnecessary_slow,
        "unnecessary_slow_coef": speed.new_tensor(
            float(unnecessary_slow_coef)
        ),
        "nominal_point_risk": nominal_point_risk,
        "slowdown_request_mean": slowdown_request_mean,
        "extra_delay": extra_delay,
        "extra_delay_coef": speed.new_tensor(float(extra_delay_coef)),
        "detour_delay": detour_delay,
        "detour_delay_coef": speed.new_tensor(float(detour_delay_coef)),
        "dual_delay": dual_delay,
        "dual_delay_coef": speed.new_tensor(float(dual_delay_coef)),
        "explicit_gap": explicit_gap,
        "explicit_gap_coef": speed.new_tensor(float(explicit_gap_coef)),
        "explicit_anchor": explicit_anchor,
        "explicit_anchor_coef": speed.new_tensor(float(explicit_anchor_coef)),
        "explicit_post": explicit_post,
        "explicit_post_coef": speed.new_tensor(float(explicit_post_coef)),
        "explicit_need_rate": explicit_need_rate,
        "explicit_assigned_gap": explicit_assigned_gap,
        "initial_plan_weight": speed.new_tensor(float(initial_plan_weight)),
        "nominal_conflict_rate": nominal_conflict.to(speed.dtype).mean(),
        "peak_collision_mode": speed.new_tensor(float(peak_collision)),
        "human_clearance": speed.new_tensor(float(human_clearance)),
        "arrival_gap_loss": arrival_gap_loss,
        "crossing_time_gap": crossing_time_gap,
        "crossing_conflict_rate": crossing_conflict.mean(),
        "arrival_gap_mode": speed.new_tensor(float(arrival_gap)),
        "arrival_time_margin": speed.new_tensor(float(arrival_time_margin)),
        "arrival_gap_coef": speed.new_tensor(float(arrival_gap_coef)),
        "fixed_yield_agent1": speed.new_tensor(float(fixed_yield_agent1)),
        "priority_speed": priority_speed,
        "bidirectional_target_gap": speed.new_tensor(float(bidirectional_target_gap)),
        "state_ordered_target_gap": speed.new_tensor(float(state_ordered_target_gap)),
        "fixed_yield_speed": speed.new_tensor(float(fixed_yield_speed)),
        "path_collision_only": speed.new_tensor(float(path_collision_only)),
        "collision_focus_steps": speed.new_tensor(float(collision_focus_steps)),
        "collision_time_uncertainty": speed.new_tensor(
            float(collision_time_uncertainty)
        ),
        "proximity_collision": raw_proximity_collision,
        "proximity_collision_mode": speed.new_tensor(float(proximity_collision)),
        "proximity_approach_beta": speed.new_tensor(
            float(proximity_approach_beta)
        ),
        "proximity_margin": speed.new_tensor(float(proximity_margin)),
        "future_collision_coef": speed.new_tensor(float(future_collision_coef)),
        "speed_efficiency": speed_efficiency,
        "speed_efficiency_coef": speed.new_tensor(float(speed_efficiency_coef)),
        "bidirectional_gap_loss": bidirectional_gap_loss,
        "profile_efficiency": profile_efficiency,
        "target_gap_need_rate": target_gap_needed.mean(),
        "target_gap_actual": target_gap_actual,
        "yield_agent1_fraction": yield_agent1_fraction,
        "prefix_speed_target_loss": prefix_speed_target_loss,
        "restore_speed_loss": restore_speed_loss,
        "target_yield_speed": target_yield_speed,
        "path_residual": path_residual,
        "path_residual_coef": speed.new_tensor(float(path_residual_coef)),
        "path_smooth": path_smooth,
        "path_smooth_coef": speed.new_tensor(float(path_smooth_coef)),
        "measured_executor_timing": speed.new_tensor(
            float(measured_executor_timing)
        ),
        "consistency": consistency["total"],
        "consistency_position": consistency["position"],
        "consistency_speed": consistency["speed"],
        "consistency_valid_fraction": consistency["valid_fraction"],
        "consistency_coef": speed.new_tensor(float(consistency_coef)),
    }
