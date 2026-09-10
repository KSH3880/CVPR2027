"""Shared-frame encoding and smooth path/speed construction."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .schema import (
    ACCEL_KNOTS,
    AGENTS,
    MAX_ACCEL,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    WAYPOINT_RESIDUAL_POINTS,
    CoordinatorState,
)


POSITION_SCALE = 10.0
HEIGHT_SCALE = 2.0
VELOCITY_SCALE = 3.0
SIZE_SCALE = 2.0
TOKEN_DIM = 12
TOKENS = AGENTS * 3


def rotate_xy(value: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    while c.ndim < value.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    x, y = value[..., 0], value[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def shared_frame(state: CoordinatorState) -> Tuple[torch.Tensor, torch.Tensor]:
    return state.root_xy.mean(dim=1), state.heading[:, 0]


def world_to_shared(points: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    while center.ndim < points.ndim:
        center = center.unsqueeze(-2)
    return rotate_xy(points - center, -angle)


def shared_to_world(points: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    result = rotate_xy(points, angle)
    while center.ndim < result.ndim:
        center = center.unsqueeze(-2)
    return result + center


def state_to_tokens(state: CoordinatorState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Six tokens in stable A.root/A.box/A.goal/B.root/B.box/B.goal order."""
    state.validate()
    center, angle = shared_frame(state)
    root = world_to_shared(state.root_xy, center, angle)
    box = world_to_shared(state.box_xyz[..., :2], center, angle)
    goal = world_to_shared(state.goal_xy, center, angle)
    root_vel = rotate_xy(state.root_vel_xy, -angle)
    box_vel = rotate_xy(state.box_vel_xy, -angle)
    rel_heading = state.heading - angle[:, None]
    rel_box_heading = state.box_heading - angle[:, None]

    tokens = torch.zeros(
        state.batch_size, AGENTS, 3, TOKEN_DIM,
        device=state.device, dtype=state.root_xy.dtype,
    )
    tokens[:, :, 0, 0:2] = root / POSITION_SCALE
    tokens[:, :, 0, 2:4] = torch.stack((torch.sin(rel_heading), torch.cos(rel_heading)), dim=-1)
    tokens[:, :, 0, 4:6] = root_vel / VELOCITY_SCALE
    tokens[:, :, 0, 6] = state.held
    tokens[:, :, 0, 7] = state.phase / 3.0

    tokens[:, :, 1, 0:2] = box / POSITION_SCALE
    tokens[:, :, 1, 2] = state.box_xyz[..., 2] / HEIGHT_SCALE
    tokens[:, :, 1, 3:5] = torch.stack(
        (torch.sin(rel_box_heading), torch.cos(rel_box_heading)), dim=-1
    )
    tokens[:, :, 1, 5:7] = box_vel / VELOCITY_SCALE
    tokens[:, :, 1, 7:9] = state.box_size_xy / SIZE_SCALE
    tokens[:, :, 1, 9] = state.held
    tokens[:, :, 1, 10] = state.phase / 3.0

    tokens[:, :, 2, 0:2] = goal / POSITION_SCALE
    tokens[:, :, 2, 6] = state.held
    tokens[:, :, 2, 7] = state.phase / 3.0
    tokens = tokens.reshape(state.batch_size, TOKENS, TOKEN_DIM)
    frame = {"center": center, "angle": angle, "root": root, "box": box, "goal": goal}
    return tokens, frame


def _bezier(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, 17, device=p0.device, dtype=p0.dtype)
    shape = (1,) * (p0.ndim - 1) + (17, 1)
    t = t.reshape(shape)
    omt = 1.0 - t
    return (
        omt ** 3 * p0.unsqueeze(-2)
        + 3.0 * omt ** 2 * t * p1.unsqueeze(-2)
        + 3.0 * omt * t ** 2 * p2.unsqueeze(-2)
        + t ** 3 * p3.unsqueeze(-2)
    )


def build_bezier_paths(frame: Dict[str, torch.Tensor], control_residual: torch.Tensor) -> torch.Tensor:
    """Create [B,K,2,33,2] paths with exact root/box/goal anchors."""
    if control_residual.ndim != 5 or control_residual.shape[2:] != (AGENTS, 4, 2):
        raise ValueError(f"expected [B,K,2,4,2], got {tuple(control_residual.shape)}")
    root = frame["root"][:, None]
    box = frame["box"][:, None]
    goal = frame["goal"][:, None]
    first_delta = box - root
    second_delta = goal - box
    base = torch.stack(
        (
            root + first_delta / 3.0,
            root + 2.0 * first_delta / 3.0,
            box + second_delta / 3.0,
            box + 2.0 * second_delta / 3.0,
        ),
        dim=-2,
    )
    control = base + control_residual
    first = _bezier(root, control[..., 0, :], control[..., 1, :], box)
    second = _bezier(box, control[..., 2, :], control[..., 3, :], goal)
    path = torch.cat((first, second[..., 1:, :]), dim=-2)
    # Reassert the public hard-anchor contract after all arithmetic.
    path[..., 0, :] = root
    path[..., 16, :] = box
    path[..., 32, :] = goal
    return path


def build_waypoint_paths(
    frame: Dict[str, torch.Tensor], waypoint_residual: torch.Tensor,
    smoothing_passes: int = 0,
    distance_scaling: bool = False,
) -> torch.Tensor:
    """Build direct 33-point polylines with exact root/box/goal anchors.

    The network predicts one shared-local offset for every non-anchor point:
    15 on root->box and 15 on box->goal. Zero is the exact piecewise-linear
    path, while both legs remain free to bend when avoidance needs it.
    """
    expected = (AGENTS, WAYPOINT_RESIDUAL_POINTS, 2)
    if waypoint_residual.ndim != 5 or waypoint_residual.shape[2:] != expected:
        raise ValueError(
            f"expected [B,K,{AGENTS},{WAYPOINT_RESIDUAL_POINTS},2], "
            f"got {tuple(waypoint_residual.shape)}"
        )
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be non-negative")
    if smoothing_passes:
        # Each 15-point leg is filtered independently with fixed zero residual
        # at root/box/goal. This preserves direct waypoint freedom while
        # removing the high-frequency zigzags that violate the 46-degree
        # executor bridge contract.
        legs = []
        for leg in (waypoint_residual[..., :15, :], waypoint_residual[..., 15:, :]):
            for _ in range(smoothing_passes):
                zero = torch.zeros_like(leg[..., :1, :])
                padded = torch.cat((zero, leg, zero), dim=-2)
                leg = (
                    0.25 * padded[..., :-2, :]
                    + 0.50 * padded[..., 1:-1, :]
                    + 0.25 * padded[..., 2:, :]
                )
            legs.append(leg)
        waypoint_residual = torch.cat(legs, dim=-2)
    root = frame["root"][:, None]
    box = frame["box"][:, None]
    goal = frame["goal"][:, None]
    if distance_scaling:
        # Residuals are metre-valued on long legs, but shrink with the
        # remaining leg below one metre. Near a hard anchor this prevents a
        # harmless-looking centimetre offset from becoming a >46-degree turn.
        first_scale = (box - root).norm(dim=-1).clamp(max=1.0)[..., None, None]
        second_scale = (goal - box).norm(dim=-1).clamp(max=1.0)[..., None, None]
        waypoint_residual = torch.cat(
            (
                waypoint_residual[..., :15, :] * first_scale,
                waypoint_residual[..., 15:, :] * second_scale,
            ),
            dim=-2,
        )
    t = torch.linspace(
        0.0, 1.0, 17, device=root.device, dtype=root.dtype
    ).reshape(1, 1, 1, 17, 1)
    first = root[..., None, :] + t * (box - root)[..., None, :]
    second = box[..., None, :] + t * (goal - box)[..., None, :]
    path = torch.cat((first, second[..., 1:, :]), dim=-2)
    path[..., 1:16, :] += waypoint_residual[..., :15, :]
    path[..., 17:32, :] += waypoint_residual[..., 15:, :]
    path[..., 0, :] = root
    path[..., 16, :] = box
    path[..., 32, :] = goal
    return path


def integrate_speed(
    path: torch.Tensor,
    accel_knots_raw: torch.Tensor,
    initial_speed: torch.Tensor,
    max_accel: float = MAX_ACCEL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate bounded acceleration knots and integrate along arc length.

    This makes abrupt per-waypoint speed jumps impossible by construction.
    """
    if path.shape[-2:] != (PATH_POINTS, 2):
        raise ValueError(f"path must end in [{PATH_POINTS},2]")
    if accel_knots_raw.shape[-1] != ACCEL_KNOTS:
        raise ValueError(f"expected {ACCEL_KNOTS} acceleration knots")
    lead = accel_knots_raw.shape[:-1]
    flat = accel_knots_raw.reshape(-1, 1, ACCEL_KNOTS)
    accel = F.interpolate(flat, size=PATH_POINTS - 1, mode="linear", align_corners=True)
    accel = max_accel * torch.tanh(accel.reshape(*lead, PATH_POINTS - 1))
    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    speed = initial_speed.clamp(MIN_SPEED, MAX_SPEED)
    values = [speed]
    for index in range(PATH_POINTS - 1):
        speed_sq = speed.square() + 2.0 * accel[..., index] * ds[..., index]
        speed = speed_sq.clamp(MIN_SPEED ** 2, MAX_SPEED ** 2).sqrt()
        values.append(speed)
    return torch.stack(values, dim=-1), accel


def direct_speed_profile(
    path: torch.Tensor,
    speed_knots_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode smooth path-local speed targets without cumulative drift.

    Zero is exactly the nominal 1.5 m/s profile. Positive knots request local
    slowdown; non-positive knots remain at nominal speed.  A centred softplus
    keeps a usable slowdown gradient at zero without the dead upper-clamp
    region of the previous signed-tanh parameterization. Runtime acceleration
    limiting remains the authority for the command actually sent to ms18.
    """
    if path.shape[-2:] != (PATH_POINTS, 2):
        raise ValueError(f"path must end in [{PATH_POINTS},2]")
    if speed_knots_raw.shape[-1] != ACCEL_KNOTS:
        raise ValueError(f"expected {ACCEL_KNOTS} speed knots")
    lead = speed_knots_raw.shape[:-1]
    flat = speed_knots_raw.reshape(-1, 1, ACCEL_KNOTS)
    raw = F.interpolate(flat, size=PATH_POINTS, mode="linear", align_corners=True)
    raw = raw.reshape(*lead, PATH_POINTS)
    centred = F.softplus(raw) - F.softplus(torch.zeros_like(raw))
    slowdown = torch.tanh(centred.clamp(min=0.0))
    speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * slowdown
    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).clamp(min=1e-4)
    acceleration = (
        (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    ).clamp(-MAX_ACCEL, MAX_ACCEL)
    return speed, acceleration


def point_speed_profile(
    path: torch.Tensor,
    point_speed_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attach one bounded speed prediction directly to every path point."""
    if path.shape[-2:] != (PATH_POINTS, 2):
        raise ValueError(f"path must end in [{PATH_POINTS},2]")
    if point_speed_raw.shape != path.shape[:-1]:
        raise ValueError(
            f"point speed must match path points {tuple(path.shape[:-1])}, "
            f"got {tuple(point_speed_raw.shape)}"
        )
    centred = F.softplus(point_speed_raw) - F.softplus(
        torch.zeros_like(point_speed_raw)
    )
    slowdown = torch.tanh(centred.clamp(min=0.0))
    speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * slowdown
    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).clamp(min=1e-4)
    acceleration = (
        (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    ).clamp(-MAX_ACCEL, MAX_ACCEL)
    return speed, acceleration


def physical_point_speed_profile(
    path: torch.Tensor,
    point_speed_raw: torch.Tensor,
    deceleration: float = 1.0,
    acceleration_limit: float = MAX_ACCEL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode learned pointwise speed caps with physical pre/post envelopes.

    The network predicts only a slowdown request at each path point. A
    backward braking pass propagates every low cap far enough upstream to be
    reachable, and a forward acceleration pass restores speed as quickly as
    the executor limit allows. Thus multiple learned risk peaks naturally
    create multiple slowdown regions without predicting centre/width events.
    """
    if path.shape[-2:] != (PATH_POINTS, 2):
        raise ValueError(f"path must end in [{PATH_POINTS},2]")
    if point_speed_raw.shape != path.shape[:-1]:
        raise ValueError(
            f"point speed must match path points {tuple(path.shape[:-1])}, "
            f"got {tuple(point_speed_raw.shape)}"
        )
    if deceleration <= 0.0 or acceleration_limit <= 0.0:
        raise ValueError("physical acceleration limits must be positive")

    slowdown_request = torch.sigmoid(point_speed_raw)
    cap = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * slowdown_request
    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).clamp(min=1e-4)

    # Reach every future cap without exceeding the permitted braking rate.
    backward = [cap[..., -1]]
    for index in range(PATH_POINTS - 2, -1, -1):
        reachable = torch.sqrt(
            backward[-1].square() + 2.0 * float(deceleration) * ds[..., index]
        )
        backward.append(torch.minimum(cap[..., index], reachable))
    brake_envelope = torch.stack(backward[::-1], dim=-1)

    # After a low cap, recover only as fast as the frozen executor can follow.
    forward = [brake_envelope[..., 0]]
    for index in range(PATH_POINTS - 1):
        reachable = torch.sqrt(
            forward[-1].square()
            + 2.0 * float(acceleration_limit) * ds[..., index]
        )
        forward.append(torch.minimum(brake_envelope[..., index + 1], reachable))
    speed = torch.stack(forward, dim=-1).clamp(MIN_SPEED, MAX_SPEED)
    acceleration = (
        (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    )
    return speed, acceleration, slowdown_request


def slowdown_window_profile(
    path: torch.Tensor,
    window_raw: torch.Tensor,
    width_max: float = 0.0,
    smooth_depth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one smooth slowdown interval per agent.

    Raw values jointly represent centre, full width and depth. ``width_max=0``
    allows the full remaining path; a positive value caps the interval in
    metres.  Zero depth is exactly the nominal-speed analytic prior.
    """
    if path.shape[-2:] != (PATH_POINTS, 2):
        raise ValueError(f"path must end in [{PATH_POINTS},2]")
    if window_raw.shape != (*path.shape[:-2], 3):
        raise ValueError(
            f"window raw must be {tuple(path.shape[:-2]) + (3,)}, "
            f"got {tuple(window_raw.shape)}"
        )
    if width_max < 0.0:
        raise ValueError("width_max must be zero(unbounded) or positive")

    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).clamp(min=1e-4)
    arc = torch.cat((torch.zeros_like(ds[..., :1]), ds.cumsum(dim=-1)), dim=-1)
    total = arc[..., -1].clamp(min=1e-3)
    centre = torch.sigmoid(window_raw[..., 0]) * total
    minimum_width = torch.minimum(total, torch.full_like(total, 0.75))
    maximum_width = (
        total if width_max == 0.0
        else torch.minimum(total, torch.full_like(total, float(width_max)))
    )
    width = minimum_width + torch.sigmoid(window_raw[..., 1]) * (
        maximum_width - minimum_width
    )
    if smooth_depth:
        # Small positive analytic start without the legacy clamp dead zone.
        # Negative raw values can still approach zero, but retain a gradient
        # that lets collision loss reactivate slowdown later.
        depth_fraction = torch.sigmoid(window_raw[..., 2] - 3.0)
    else:
        centred_depth = F.softplus(window_raw[..., 2]) - F.softplus(
            torch.zeros_like(window_raw[..., 2])
        )
        depth_fraction = torch.tanh(centred_depth.clamp(min=0.0))
    depth = (MAX_SPEED - MIN_SPEED) * depth_fraction

    relative = (arc - centre[..., None]).abs() / (
        0.5 * width[..., None]
    ).clamp(min=1e-3)
    inside = relative < 1.0
    bump = 0.5 * (1.0 + torch.cos(torch.pi * relative.clamp(max=1.0)))
    bump = torch.where(inside, bump, torch.zeros_like(bump))
    speed = (MAX_SPEED - depth[..., None] * bump).clamp(MIN_SPEED, MAX_SPEED)
    acceleration = (
        (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    ).clamp(-MAX_ACCEL, MAX_ACCEL)
    decoded = torch.stack((centre, width, depth), dim=-1)
    return speed, acceleration, decoded


def conflict_window_profile(
    path: torch.Tensor,
    window_raw: torch.Tensor,
    straight_path: torch.Tensor,
    smooth_depth: bool = False,
    min_at_crossing: bool = False,
    plateau_before_crossing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place one learned slowdown interval immediately before route crossing.

    The joint network emits ``(length, depth)`` for both agents together.  The
    interval end is not learned.  For each possible yielder it is the closest
    point between its predicted path and the other agent's hard straight
    priority path.  The legacy profile uses a symmetric cosine that returns to
    nominal speed at that point.  ``min_at_crossing`` instead reaches minimum
    speed there and deterministically accelerates afterward.  This keeps path
    and speed in one joint output while removing the poorly identified free
    centre/width parameterization.
    """
    if path.shape[-3:] != (AGENTS, PATH_POINTS, 2):
        raise ValueError(
            f"path must end in [{AGENTS},{PATH_POINTS},2], got {tuple(path.shape)}"
        )
    if window_raw.shape != (*path.shape[:-3], AGENTS, 2):
        raise ValueError(
            f"window raw must be {tuple(path.shape[:-3]) + (AGENTS, 2)}, "
            f"got {tuple(window_raw.shape)}"
        )
    if straight_path.shape != path.shape:
        raise ValueError(
            f"straight path must match path {tuple(path.shape)}, "
            f"got {tuple(straight_path.shape)}"
        )

    ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).clamp(min=1e-4)
    arc = torch.cat((torch.zeros_like(ds[..., :1]), ds.cumsum(dim=-1)), dim=-1)

    # Agent 0 as yielder is compared with agent 1's fixed priority route, and
    # vice versa. This matches apply_fixed_priority even though role is not a
    # model input and both counterfactual yield plans are emitted together.
    pair0 = torch.cdist(path[..., 0, :, :], straight_path[..., 1, :, :])
    pair1 = torch.cdist(straight_path[..., 0, :, :], path[..., 1, :, :])
    closest0 = pair0.flatten(start_dim=-2).argmin(dim=-1)
    closest1 = pair1.flatten(start_dim=-2).argmin(dim=-1)
    index0 = torch.div(closest0, PATH_POINTS, rounding_mode="floor")
    index1 = closest1 % PATH_POINTS
    cross_s = torch.stack(
        (
            arc[..., 0, :].gather(-1, index0[..., None]).squeeze(-1),
            arc[..., 1, :].gather(-1, index1[..., None]).squeeze(-1),
        ),
        dim=-1,
    )

    # Length may cover any part of the already remaining prefix, but can never
    # start behind the current root.  Zero raw depth is exactly the analytic
    # 1.5 m/s prior used by all C2 comparisons.
    minimum_length = torch.minimum(cross_s, torch.full_like(cross_s, 0.75))
    length = (
        torch.sigmoid(window_raw[..., 0]) * cross_s
        if plateau_before_crossing
        else minimum_length + torch.sigmoid(window_raw[..., 0]) * (
            cross_s - minimum_length
        )
    )
    if smooth_depth:
        depth_fraction = torch.sigmoid(window_raw[..., 1] - 3.0)
    else:
        centred_depth = F.softplus(window_raw[..., 1]) - F.softplus(
            torch.zeros_like(window_raw[..., 1])
        )
        depth_fraction = torch.tanh(centred_depth.clamp(min=0.0))
    depth = (MAX_SPEED - MIN_SPEED) * depth_fraction

    start = cross_s - length
    phase = (arc - start[..., None]) / length[..., None].clamp(min=1e-3)
    if plateau_before_crossing:
        # C6: the learned length is a constant-minimum-speed plateau ending at
        # the conflict point. Braking and recovery distances are derived from
        # the same physical acceleration limit rather than learned separately.
        min_speed = MAX_SPEED - depth
        plateau_start = cross_s - length
        distance_before = (plateau_start[..., None] - arc).clamp(min=0.0)
        braking_speed = torch.sqrt(
            min_speed[..., None].square()
            + 2.0 * MAX_ACCEL * distance_before
        ).clamp(max=MAX_SPEED)
        distance_after = (arc - cross_s[..., None]).clamp(min=0.0)
        recovery_speed = torch.sqrt(
            min_speed[..., None].square()
            + 2.0 * MAX_ACCEL * distance_after
        ).clamp(max=MAX_SPEED)
        speed = torch.where(
            arc < plateau_start[..., None], braking_speed,
            torch.where(
                arc <= cross_s[..., None], min_speed[..., None],
                recovery_speed,
            ),
        )
    elif min_at_crossing:
        # Approach at full speed, monotonically brake to the learned minimum
        # exactly at the conflict point, then recover at the same 0.75 m/s^2
        # acceleration limit used by the runtime command rate limiter.  The
        # model still emits only (pre_length, depth); recovery is deterministic.
        approach_u = phase.clamp(0.0, 1.0)
        approach_bump = approach_u.square() * (3.0 - 2.0 * approach_u)
        min_speed = MAX_SPEED - depth
        recovery_length = (
            (MAX_SPEED ** 2 - min_speed.square()) / (2.0 * MAX_ACCEL)
        ).clamp(min=1e-3)
        recovery_u = (
            (arc - cross_s[..., None]) / recovery_length[..., None]
        ).clamp(0.0, 1.0)
        recovery_bump = 1.0 - recovery_u.square() * (3.0 - 2.0 * recovery_u)
        bump = torch.where(
            arc <= cross_s[..., None], approach_bump, recovery_bump
        )
    else:
        inside = (phase > 0.0) & (phase < 1.0)
        bump = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * phase.clamp(0.0, 1.0)))
        bump = torch.where(inside, bump, torch.zeros_like(bump))
    if not plateau_before_crossing:
        speed = (MAX_SPEED - depth[..., None] * bump).clamp(MIN_SPEED, MAX_SPEED)
    acceleration = (
        (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    ).clamp(-MAX_ACCEL, MAX_ACCEL)
    decoded = torch.stack((start, cross_s, length, depth), dim=-1)
    return speed, acceleration, decoded


def _sample_polyline(values: torch.Tensor, arc: torch.Tensor) -> torch.Tensor:
    """Sample [...,N,D] values at [...,Q] arc coordinates."""
    seg = values[..., 1:, :] - values[..., :-1, :]
    length = seg.norm(dim=-1).clamp(min=1e-7)
    cumulative = torch.cat((torch.zeros_like(length[..., :1]), length.cumsum(dim=-1)), dim=-1)
    flat_cum = cumulative.reshape(-1, cumulative.shape[-1]).contiguous()
    flat_arc = arc.reshape(-1, arc.shape[-1]).contiguous()
    index = torch.searchsorted(flat_cum, flat_arc, right=True).sub(1).clamp(0, values.shape[-2] - 2)
    start = flat_cum.gather(1, index)
    flat_len = length.reshape(-1, length.shape[-1])
    frac = ((flat_arc - start) / flat_len.gather(1, index)).clamp(0.0, 1.0)
    flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
    row = torch.arange(flat_values.shape[0], device=values.device)[:, None]
    result = flat_values[row, index] + frac[..., None] * (
        flat_values[row, index + 1] - flat_values[row, index]
    )
    return result.reshape(*arc.shape, values.shape[-1])


def resample_path_and_speed(
    path: torch.Tensor,
    speed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert selected [B,2,33] plans to ms18's 0.1m/320-cell buffers."""
    if path.ndim != 4 or path.shape[1:] != (AGENTS, PATH_POINTS, 2):
        raise ValueError(f"expected path [B,2,33,2], got {tuple(path.shape)}")
    if speed.shape != path.shape[:-1]:
        raise ValueError("speed shape must equal path shape without xy")
    length = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    cumulative = torch.cat((torch.zeros_like(length[..., :1]), length.cumsum(dim=-1)), dim=-1)
    end_s = cumulative[..., -1]
    query = torch.arange(PATH_VERTICES, device=path.device, dtype=path.dtype) * PATH_DS
    query = torch.minimum(query.reshape(1, 1, -1), end_s[..., None])
    dense_path = _sample_polyline(path, query)
    speed_value = speed[..., None]
    dense_speed = _sample_polyline(speed_value, query)[..., 0].clamp(MIN_SPEED, MAX_SPEED)
    return dense_path, dense_speed, end_s
