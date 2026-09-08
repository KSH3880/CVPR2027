"""Stochastic high-level policy view of :class:`JointCoordinator` for PPO."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .geometry import (
    build_bezier_paths, build_waypoint_paths, conflict_window_profile, direct_speed_profile, integrate_speed,
    physical_point_speed_profile, point_speed_profile, slowdown_window_profile, shared_to_world,
    state_to_tokens,
)
from .model import JointCoordinator
from .schema import (
    ACCEL_KNOTS, AGENTS, CANDIDATES, MAX_SPEED, MIN_SPEED, PATH_POINTS,
    WAYPOINT_RESIDUAL_POINTS, CoordinatorState,
)


PATH_ACTIONS = CANDIDATES * AGENTS * 4 * 2
ACCEL_ACTIONS = CANDIDATES * AGENTS * ACCEL_KNOTS
DWELL_ACTIONS = CANDIDATES * AGENTS
ACTION_DIM = PATH_ACTIONS + ACCEL_ACTIONS + DWELL_ACTIONS


def _action_sizes(model: JointCoordinator) -> Tuple[int, int, int, int]:
    candidates = model.config.candidates
    path_points = (
        WAYPOINT_RESIDUAL_POINTS if model.config.direct_waypoints else 4
    )
    path = candidates * AGENTS * path_points * 2
    speed_points = (
        PATH_POINTS if model.config.joint_point_speed
        else 3 if model.config.slowdown_window
        else 2 if model.config.conflict_window
        else ACCEL_KNOTS
    )
    accel = candidates * AGENTS * speed_points
    dwell = 0 if model.config.fixed_pickup_dwell is not None else candidates * AGENTS
    return path, accel, dwell, path + accel + dwell


def pack_mean(model: JointCoordinator, output: Dict[str, torch.Tensor]) -> torch.Tensor:
    path_value = (
        output["waypoint_residual"]
        if model.config.direct_waypoints else output["control_residual"]
    )
    speed_value = (
        output["point_speed_raw"] if model.config.joint_point_speed
        else output["slowdown_window_raw"] if (model.config.slowdown_window or model.config.conflict_window)
        else output["accel_knots_raw"]
    )
    values = [path_value.flatten(start_dim=1), speed_value.flatten(start_dim=1)]
    if model.config.fixed_pickup_dwell is None:
        values.append(output["pickup_dwell"].flatten(start_dim=1))
    return torch.cat(values, dim=-1)


def decode_action(
    model: JointCoordinator,
    state: CoordinatorState,
    action: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    path_actions, accel_actions, dwell_actions, action_dim = _action_sizes(model)
    candidates = model.config.candidates
    if action.shape != (state.batch_size, action_dim):
        raise ValueError(f"expected action [B,{action_dim}], got {tuple(action.shape)}")
    start = 0
    path_value = action[:, start:start + path_actions]
    start += path_actions
    speed_points = (
        PATH_POINTS if model.config.joint_point_speed
        else 3 if model.config.slowdown_window
        else 2 if model.config.conflict_window
        else ACCEL_KNOTS
    )
    accel_raw = action[:, start:start + accel_actions].reshape(
        state.batch_size, candidates, AGENTS, speed_points
    )
    start += accel_actions
    if dwell_actions:
        dwell = action[:, start:].reshape(state.batch_size, candidates, AGENTS)
        dwell = dwell.clamp(0.5, 3.0)
    else:
        dwell = template["pickup_dwell"]
    waypoint_residual = None
    if model.config.direct_waypoints:
        waypoint_residual = path_value.reshape(
            state.batch_size, candidates, AGENTS,
            WAYPOINT_RESIDUAL_POINTS, 2,
        ).clamp(-model.config.residual_scale, model.config.residual_scale)
        waypoint_residual = waypoint_residual.clone()
        held = state.held[:, None, :, None, None] >= 0.5
        waypoint_residual[..., :15, :] = torch.where(
            held, torch.zeros_like(waypoint_residual[..., :15, :]),
            waypoint_residual[..., :15, :],
        )
        residual = torch.zeros(
            state.batch_size, candidates, AGENTS, 4, 2,
            device=state.device, dtype=state.root_xy.dtype,
        )
    else:
        residual = path_value.reshape(
            state.batch_size, candidates, AGENTS, 4, 2
        ).clamp(-model.config.residual_scale, model.config.residual_scale)
        if model.config.analytic_prior:
            # Exploration must not corrupt the pickup approach. PPO only owns
            # the carry-leg correction in the Bezier C2 configuration.
            residual = residual.clone()
            residual[..., :2, :] = 0.0
    accel_raw = accel_raw.clamp(-5.0, 5.0)
    dwell = torch.where(state.held[:, None] >= 0.5, torch.zeros_like(dwell), dwell)

    _, frame = state_to_tokens(state)
    path_local = (
        build_waypoint_paths(
            frame, waypoint_residual,
            model.config.waypoint_smoothing_passes,
            model.config.waypoint_distance_scaling,
        )
        if model.config.direct_waypoints
        else build_bezier_paths(frame, residual)
    )
    path_world = shared_to_world(
        path_local, frame["center"][:, None], frame["angle"][:, None]
    )
    slowdown_window = None
    slowdown_request = None
    if model.config.joint_point_speed:
        if model.config.physical_speed_caps:
            speed, acceleration, slowdown_request = physical_point_speed_profile(
                path_world, accel_raw
            )
        else:
            speed, acceleration = point_speed_profile(path_world, accel_raw)
    elif model.config.slowdown_window:
        speed, acceleration, slowdown_window = slowdown_window_profile(
            path_world, accel_raw, model.config.slowdown_width_max,
            model.config.smooth_depth,
        )
    elif model.config.conflict_window:
        zero_residual = torch.zeros_like(waypoint_residual)
        straight_local = build_waypoint_paths(frame, zero_residual)
        straight_world = shared_to_world(
            straight_local, frame["center"][:, None], frame["angle"][:, None]
        )
        speed, acceleration, slowdown_window = conflict_window_profile(
            path_world, accel_raw, straight_world, model.config.smooth_depth,
            model.config.conflict_min_at_crossing,
            model.config.conflict_plateau,
        )
    elif model.config.direct_speed_profile:
        speed, acceleration = direct_speed_profile(path_world, accel_raw)
    elif model.config.immediate_slowdown:
        initial_speed = (
            MAX_SPEED
            + (MAX_SPEED - MIN_SPEED) * torch.tanh(accel_raw[..., 0])
        ).clamp(MIN_SPEED, MAX_SPEED)
    elif model.config.analytic_prior and not model.config.actual_initial_speed:
        initial_speed = torch.full(
            (state.batch_size, candidates, AGENTS),
            MAX_SPEED,
            device=state.device,
            dtype=state.root_xy.dtype,
        )
    else:
        initial_speed = state.root_vel_xy.norm(dim=-1)[:, None].expand(-1, candidates, -1)
    if not (
        model.config.direct_speed_profile
        or model.config.joint_point_speed
        or model.config.slowdown_window
        or model.config.conflict_window
    ):
        speed, acceleration = integrate_speed(path_world, accel_raw, initial_speed)
    result = {
        "path_local": path_local,
        "path_world": path_world,
        "speed": speed,
        "acceleration": acceleration,
        "pickup_dwell": dwell,
        "candidate_value": template["candidate_value"],
        "risk_logits": template["risk_logits"],
        "control_residual": residual,
        "accel_knots_raw": accel_raw,
        "trajectory": torch.cat((path_world, speed.unsqueeze(-1)), dim=-1),
    }
    if "priority_logits" in template:
        result["priority_logits"] = template["priority_logits"]
    if waypoint_residual is not None:
        result["waypoint_residual"] = waypoint_residual
    if model.config.joint_point_speed:
        result["point_speed_raw"] = accel_raw
    if slowdown_request is not None:
        result["slowdown_request"] = slowdown_request
    if model.config.slowdown_window or model.config.conflict_window:
        result["slowdown_window_raw"] = accel_raw
        result["slowdown_window"] = slowdown_window
    return result


class CoordinatorActorCritic(nn.Module):
    """PPO distribution over all K unnamed joint futures."""

    def __init__(self, coordinator: Optional[JointCoordinator] = None, init_std: float = 0.15):
        super().__init__()
        self.coordinator = coordinator or JointCoordinator()
        path_actions, _, _, self.action_dim = _action_sizes(self.coordinator)
        initial_std = torch.full((self.action_dim,), float(init_std))
        if self.coordinator.config.direct_waypoints:
            # Independent 15 cm noise at all 120 waypoint coordinates creates
            # a jagged invalid polyline before PPO gets any useful rollout.
            # Keep meaningful speed exploration while perturbing path points
            # by only 5 mm around the exact straight prior. Some pickup legs
            # are ~10 cm per segment, where even 2 cm iid noise can violate
            # the 46-degree bridge curvature contract.
            initial_std[:path_actions] = 0.005
        self.action_log_std = nn.Parameter(
            initial_std.log()
        )

    def distribution(self, state: CoordinatorState) -> Tuple[Normal, torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.coordinator(state)
        mean = pack_mean(self.coordinator, output)
        minimum_std = 0.001 if self.coordinator.config.direct_waypoints else 0.01
        std = self.action_log_std.exp().clamp(minimum_std, 1.0).expand_as(mean)
        value = output["candidate_value"].mean(dim=1)
        return Normal(mean, std), value, output

    def act(self, state: CoordinatorState, deterministic: bool = False):
        distribution, value, template = self.distribution(state)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        output = decode_action(self.coordinator, state, action, template)
        return output, action, log_prob, value

    def evaluate(self, state: CoordinatorState, action: torch.Tensor):
        distribution, value, output = self.distribution(state)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value, output
