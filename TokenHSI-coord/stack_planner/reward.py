"""Outcome-based macro reward for the isolated stack planner.

The reward never reads a GT path, retreat direction, virtual box, or legacy
Carry observation.  It consumes physical measurements before and after one
planner macro action plus costs accumulated while the frozen executor runs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

import torch


@dataclass(frozen=True)
class StackRewardConfig:
    # Task potential: bottom -> clearance -> top.  Products make later terms
    # valuable only after their physical dependencies are satisfied.
    bottom_weight: float = 1.0
    clearance_weight: float = 1.0
    top_weight: float = 2.0

    bottom_xy_coefficient: float = 12.0
    bottom_z_coefficient: float = 40.0
    bottom_linear_speed_coefficient: float = 4.0
    bottom_angular_speed_coefficient: float = 0.5
    bottom_tilt_coefficient: float = 8.0

    top_xy_coefficient: float = 50.0
    top_z_coefficient: float = 40.0
    top_linear_speed_coefficient: float = 4.0
    top_angular_speed_coefficient: float = 0.5
    top_tilt_coefficient: float = 8.0

    clearance_start: float = 0.6
    clearance_done: float = 1.5
    released_floor: float = 0.5
    approach_coefficient: float = 2.0
    approach_weight: float = 0.25

    success_bonus: float = 5.0
    collision_weight: float = 1.0
    bottom_disturbance_weight: float = 0.5
    humanoid_fall_penalty: float = 3.0
    time_penalty_per_second: float = 0.02
    invalid_plan_penalty: float = 0.25
    unsafe_plan_penalty: float = 0.25

    def __post_init__(self) -> None:
        non_negative = (
            "bottom_weight", "clearance_weight", "top_weight",
            "bottom_xy_coefficient", "bottom_z_coefficient",
            "bottom_linear_speed_coefficient",
            "bottom_angular_speed_coefficient", "bottom_tilt_coefficient",
            "top_xy_coefficient", "top_z_coefficient",
            "top_linear_speed_coefficient",
            "top_angular_speed_coefficient", "top_tilt_coefficient",
            "success_bonus", "collision_weight",
            "bottom_disturbance_weight", "humanoid_fall_penalty",
            "time_penalty_per_second",
            "invalid_plan_penalty", "unsafe_plan_penalty",
            "approach_coefficient", "approach_weight",
        )
        for name in non_negative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.clearance_done <= self.clearance_start:
            raise ValueError("clearance_done must exceed clearance_start")
        if not 0.0 <= self.released_floor <= 1.0:
            raise ValueError("released_floor must be in [0, 1]")


@dataclass
class StackPhysicalState:
    """Physical task measurements at one planner decision boundary.

    Position errors are measured against the actual bottom/top placement
    constraints.  ``clearance`` is direction-free: it can be root-to-workspace
    distance for a first smoke test, or minimum non-hand body clearance for a
    stricter experiment.
    """

    bottom_position_error: torch.Tensor       # [B, 3]
    bottom_linear_velocity: torch.Tensor      # [B, 3]
    bottom_angular_velocity: torch.Tensor     # [B, 3]
    bottom_tilt: torch.Tensor                 # [B], radians or a consistent proxy
    bottom_released: torch.Tensor             # [B], 0/1 physical contact state
    bottom_root_distance: torch.Tensor        # [B], metres to actual bottom box
    clearance: torch.Tensor                   # [B], metres
    top_position_error: torch.Tensor          # [B, 3]
    top_linear_velocity: torch.Tensor         # [B, 3]
    top_angular_velocity: torch.Tensor        # [B, 3]
    top_tilt: torch.Tensor                    # [B]
    top_released: torch.Tensor                # [B], 0/1 physical contact state
    top_root_distance: torch.Tensor           # [B], metres to actual top box
    stack_success: torch.Tensor               # [B], 0/1 stable terminal condition

    def validate(self) -> None:
        vector_names = (
            "bottom_position_error", "bottom_linear_velocity",
            "bottom_angular_velocity", "top_position_error",
            "top_linear_velocity", "top_angular_velocity",
        )
        batch = None
        reference = None
        for item in fields(self):
            value = getattr(self, item.name)
            if not torch.is_tensor(value):
                raise TypeError(f"{item.name} must be a tensor")
            expected_ndim = 2 if item.name in vector_names else 1
            if value.ndim != expected_ndim:
                raise ValueError(
                    f"{item.name} must have rank {expected_ndim}, got {value.ndim}"
                )
            if item.name in vector_names and value.shape[1] != 3:
                raise ValueError(f"{item.name} must have shape [B,3]")
            if batch is None:
                batch, reference = value.shape[0], value
            elif value.shape[0] != batch:
                raise ValueError("all physical state tensors must share batch size")
            if value.device != reference.device:
                raise ValueError("all physical state tensors must share device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{item.name} contains non-finite values")
        for name in ("bottom_released", "top_released", "stack_success"):
            value = getattr(self, name)
            if ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("clearance", "bottom_root_distance", "top_root_distance"):
            if (getattr(self, name) < 0).any():
                raise ValueError(f"{name} must be non-negative")

    @property
    def batch_size(self) -> int:
        return self.bottom_tilt.shape[0]


@dataclass
class StackIntervalCosts:
    """Measurements accumulated while one planner trajectory is executed."""

    collision: torch.Tensor             # [B], mean/max physical overlap cost
    bottom_disturbance: torch.Tensor     # [B], displacement/motion accumulated
    humanoid_fall: torch.Tensor          # [B], any-agent fall latched to 0/1
    elapsed_seconds: torch.Tensor        # [B]
    invalid_plan: torch.Tensor           # [B], 0/1
    unsafe_plan: torch.Tensor            # [B], 0/1

    def validate(self, batch_size: int, device: torch.device) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not torch.is_tensor(value):
                raise TypeError(f"{item.name} must be a tensor")
            if value.shape != (batch_size,):
                raise ValueError(
                    f"{item.name} must have shape [{batch_size}], got {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError("all interval costs must share physical state device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{item.name} contains non-finite values")
            if (value < 0).any():
                raise ValueError(f"{item.name} must be non-negative")
        for name in ("humanoid_fall", "invalid_plan", "unsafe_plan"):
            value = getattr(self, name)
            if (value > 1).any():
                raise ValueError(f"{name} must be in [0, 1]")


def _placement_quality(
    position_error: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    tilt: torch.Tensor,
    released: torch.Tensor,
    *,
    xy_coefficient: float,
    z_coefficient: float,
    linear_speed_coefficient: float,
    angular_speed_coefficient: float,
    tilt_coefficient: float,
    released_floor: float,
) -> torch.Tensor:
    xy_error2 = position_error[..., :2].square().sum(dim=-1)
    z_error2 = position_error[..., 2].square()
    linear_speed2 = linear_velocity.square().sum(dim=-1)
    angular_speed2 = angular_velocity.square().sum(dim=-1)
    cost = (
        xy_coefficient * xy_error2
        + z_coefficient * z_error2
        + linear_speed_coefficient * linear_speed2
        + angular_speed_coefficient * angular_speed2
        + tilt_coefficient * tilt.square()
    )
    # Keep a dense approach/placement signal before release, then make a clean
    # release strictly better at the same pose and stability.
    release_score = released_floor + (1.0 - released_floor) * released.to(
        position_error.dtype
    )
    # Unlike exp(-cost), this retains measurable progress several metres from
    # the placement target.  It is still a physical outcome potential, not a
    # path or retreat-direction target.
    return release_score / (1.0 + cost)


def stack_task_potential(
    state: StackPhysicalState,
    config: StackRewardConfig = StackRewardConfig(),
) -> Dict[str, torch.Tensor]:
    """Compute direction-free, dependency-ordered physical task quality."""
    state.validate()
    bottom_placement = _placement_quality(
        state.bottom_position_error,
        state.bottom_linear_velocity,
        state.bottom_angular_velocity,
        state.bottom_tilt,
        state.bottom_released,
        xy_coefficient=config.bottom_xy_coefficient,
        z_coefficient=config.bottom_z_coefficient,
        linear_speed_coefficient=config.bottom_linear_speed_coefficient,
        angular_speed_coefficient=config.bottom_angular_speed_coefficient,
        tilt_coefficient=config.bottom_tilt_coefficient,
        released_floor=config.released_floor,
    )
    bottom_approach = 1.0 / (
        1.0 + config.approach_coefficient * state.bottom_root_distance.square()
    )
    bottom = (
        bottom_placement
        + config.approach_weight * (1.0 - bottom_placement) * bottom_approach
    )
    clearance = (
        (state.clearance - config.clearance_start)
        / (config.clearance_done - config.clearance_start)
    ).clamp(0.0, 1.0)
    top_placement = _placement_quality(
        state.top_position_error,
        state.top_linear_velocity,
        state.top_angular_velocity,
        state.top_tilt,
        state.top_released,
        xy_coefficient=config.top_xy_coefficient,
        z_coefficient=config.top_z_coefficient,
        linear_speed_coefficient=config.top_linear_speed_coefficient,
        angular_speed_coefficient=config.top_angular_speed_coefficient,
        tilt_coefficient=config.top_tilt_coefficient,
        released_floor=config.released_floor,
    )
    top_approach = 1.0 / (
        1.0 + config.approach_coefficient * state.top_root_distance.square()
    )
    top = (
        top_placement
        + config.approach_weight * (1.0 - top_placement) * top_approach
    )
    bottom_and_clear = bottom * clearance
    full_stack = bottom_and_clear * top
    potential = (
        config.bottom_weight * bottom
        + config.clearance_weight * bottom_and_clear
        + config.top_weight * full_stack
    )
    return {
        "potential": potential,
        "bottom_quality": bottom,
        "bottom_placement_quality": bottom_placement,
        "bottom_approach_quality": bottom_approach,
        "clearance_quality": clearance,
        "top_quality": top,
        "top_placement_quality": top_placement,
        "top_approach_quality": top_approach,
        "bottom_and_clear": bottom_and_clear,
        "full_stack_quality": full_stack,
    }


def compute_stack_planner_reward(
    previous: StackPhysicalState,
    current: StackPhysicalState,
    interval: StackIntervalCosts,
    config: StackRewardConfig = StackRewardConfig(),
) -> Dict[str, torch.Tensor]:
    """Reward one planner macro action from its actual executor outcome.

    Call this after installing a plan and running the frozen low-level agent
    for N steps.  All returned values have shape ``[B]`` and can be logged
    independently; ``total`` is the PPO reward for that planner transition.
    """
    previous_quality = stack_task_potential(previous, config)
    current_quality = stack_task_potential(current, config)
    if previous.batch_size != current.batch_size:
        raise ValueError("previous and current states must share batch size")
    interval.validate(current.batch_size, current.bottom_tilt.device)

    potential_delta = (
        current_quality["potential"] - previous_quality["potential"]
    )
    new_success = (
        current.stack_success.to(current.bottom_tilt.dtype)
        - previous.stack_success.to(current.bottom_tilt.dtype)
    ).clamp(0.0, 1.0)
    success = config.success_bonus * new_success
    collision = -config.collision_weight * interval.collision
    disturbance = (
        -config.bottom_disturbance_weight * interval.bottom_disturbance
    )
    fall = -config.humanoid_fall_penalty * interval.humanoid_fall.to(
        current.bottom_tilt.dtype
    )
    time = -config.time_penalty_per_second * interval.elapsed_seconds
    invalid = -config.invalid_plan_penalty * interval.invalid_plan.to(
        current.bottom_tilt.dtype
    )
    unsafe = -config.unsafe_plan_penalty * interval.unsafe_plan.to(
        current.bottom_tilt.dtype
    )
    total = (
        potential_delta + success + collision + disturbance + fall + time
        + invalid + unsafe
    )
    return {
        "total": total,
        "potential_delta": potential_delta,
        "success_bonus": success,
        "collision_penalty": collision,
        "bottom_disturbance_penalty": disturbance,
        "humanoid_fall_penalty": fall,
        "time_penalty": time,
        "invalid_plan_penalty": invalid,
        "unsafe_plan_penalty": unsafe,
        **{f"current_{key}": value for key, value in current_quality.items()},
    }


__all__ = [
    "StackIntervalCosts",
    "StackPhysicalState",
    "StackRewardConfig",
    "compute_stack_planner_reward",
    "stack_task_potential",
]
