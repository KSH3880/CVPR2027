"""Strict checkpoint contract for the C2 K=1 residual coordinator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .model import CoordinatorConfig, JointCoordinator
from .schema import (
    ACCEL_KNOTS,
    AGENTS,
    MAX_ACCEL,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    STEER_HORIZON_SECONDS,
    STEER_POINTS,
    WAYPOINT_RESIDUAL_POINTS,
)


C2_SCHEMA_VERSION = "tokenhsi-coord-c2-v1"
C2_CANDIDATES = 1
C2_FIXED_DWELL = 1.5
C2_ACTION_DIM = C2_CANDIDATES * AGENTS * (4 * 2 + ACCEL_KNOTS)
C2_WAYPOINT_ACTION_DIM = C2_CANDIDATES * AGENTS * (
    WAYPOINT_RESIDUAL_POINTS * 2 + ACCEL_KNOTS
)
C2_JOINT_POINT_ACTION_DIM = C2_CANDIDATES * AGENTS * (
    WAYPOINT_RESIDUAL_POINTS * 2 + PATH_POINTS
)
C2_SLOWDOWN_WINDOW_ACTION_DIM = C2_CANDIDATES * AGENTS * (
    WAYPOINT_RESIDUAL_POINTS * 2 + 3
)
C2_CONFLICT_WINDOW_ACTION_DIM = C2_CANDIDATES * AGENTS * (
    WAYPOINT_RESIDUAL_POINTS * 2 + 2
)


def c2_config(
    *, actual_initial_speed: bool = False, immediate_slowdown: bool = False,
    direct_speed_profile: bool = False,
    arrival_features: bool = False,
    mlp_backbone: bool = False,
    mlp_conflict_features: bool = False,
    direct_waypoints: bool = False,
    joint_point_speed: bool = False,
    physical_speed_caps: bool = False,
    waypoint_smoothing_passes: int = 0,
    waypoint_distance_scaling: bool = False,
    slowdown_window: bool = False,
    slowdown_width_max: float = 0.0,
    conflict_window: bool = False,
    conflict_min_at_crossing: bool = False,
    conflict_plateau: bool = False,
    smooth_depth: bool = False,
    learned_priority: bool = False,
) -> CoordinatorConfig:
    return CoordinatorConfig(
        candidates=C2_CANDIDATES,
        residual_scale=1.0,
        analytic_prior=True,
        fixed_pickup_dwell=C2_FIXED_DWELL,
        actual_initial_speed=actual_initial_speed,
        immediate_slowdown=immediate_slowdown,
        direct_speed_profile=direct_speed_profile,
        arrival_features=arrival_features,
        mlp_backbone=mlp_backbone,
        mlp_conflict_features=mlp_conflict_features,
        direct_waypoints=direct_waypoints,
        joint_point_speed=joint_point_speed,
        physical_speed_caps=physical_speed_caps,
        waypoint_smoothing_passes=waypoint_smoothing_passes,
        waypoint_distance_scaling=waypoint_distance_scaling,
        slowdown_window=slowdown_window,
        slowdown_width_max=slowdown_width_max,
        conflict_window=conflict_window,
        conflict_min_at_crossing=conflict_min_at_crossing,
        conflict_plateau=conflict_plateau,
        smooth_depth=smooth_depth,
        learned_priority=learned_priority,
    )


def _valid_c2_config(config: CoordinatorConfig) -> bool:
    reference = c2_config()
    fixed_fields = (
        "token_dim", "d_model", "nhead", "encoder_layers", "decoder_layers",
        "feedforward", "dropout", "candidates", "residual_scale",
        "analytic_prior", "fixed_pickup_dwell",
    )
    fixed_ok = all(getattr(config, key) == getattr(reference, key) for key in fixed_fields)
    speed_modes = sum((
        bool(config.actual_initial_speed), bool(config.immediate_slowdown),
        bool(config.direct_speed_profile),
    ))
    waypoint_ok = not config.direct_waypoints or config.mlp_backbone
    conflict_features_ok = (
        not config.mlp_conflict_features or config.mlp_backbone
    )
    joint_ok = not config.joint_point_speed or (
        config.mlp_backbone
        and config.direct_waypoints
        and config.direct_speed_profile
    )
    physical_caps_ok = not config.physical_speed_caps or config.joint_point_speed
    window_ok = (
        config.mlp_backbone
        and config.direct_waypoints
        and config.direct_speed_profile
        and not config.joint_point_speed
        and config.slowdown_width_max >= 0.0
    ) if config.slowdown_window else config.slowdown_width_max == 0.0
    conflict_window_ok = (
        config.mlp_backbone
        and config.direct_waypoints
        and config.direct_speed_profile
        and not config.joint_point_speed
        and not config.slowdown_window
    ) if config.conflict_window else True
    conflict_profile_ok = (
        not config.conflict_min_at_crossing or config.conflict_window
    )
    conflict_plateau_ok = (
        (not config.conflict_plateau or config.conflict_window)
        and not (config.conflict_plateau and config.conflict_min_at_crossing)
    )
    smooth_depth_ok = (
        not config.smooth_depth
        or config.slowdown_window
        or config.conflict_window
    )
    smoothing_ok = (
        config.waypoint_smoothing_passes >= 0
        and (config.waypoint_smoothing_passes == 0 or config.direct_waypoints)
        and (not config.waypoint_distance_scaling or config.direct_waypoints)
    )
    return (
        fixed_ok and speed_modes <= 1 and waypoint_ok and joint_ok
        and physical_caps_ok
        and smoothing_ok and conflict_features_ok and window_ok
        and conflict_window_ok
        and conflict_profile_ok
        and conflict_plateau_ok
        and smooth_depth_ok
    )


def expected_c2_contract(
    config: Optional[CoordinatorConfig] = None,
) -> Dict[str, Any]:
    config = config or c2_config()
    action_dim = (
        C2_JOINT_POINT_ACTION_DIM if config.joint_point_speed else
        C2_SLOWDOWN_WINDOW_ACTION_DIM if config.slowdown_window else
        C2_CONFLICT_WINDOW_ACTION_DIM if config.conflict_window else
        C2_WAYPOINT_ACTION_DIM if config.direct_waypoints else C2_ACTION_DIM
    )
    contract = {
        "schema_version": C2_SCHEMA_VERSION,
        "model_kind": "c2_single_residual_transformer",
        "agents": AGENTS,
        "candidate_k": C2_CANDIDATES,
        "action_dim": action_dim,
        "path_points": PATH_POINTS,
        "accel_knots": (
            PATH_POINTS if config.joint_point_speed else
            2 if config.conflict_window else ACCEL_KNOTS
        ),
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "steer_points": STEER_POINTS,
        "steer_horizon_seconds": STEER_HORIZON_SECONDS,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
        "max_accel": MAX_ACCEL,
        "fixed_pickup_dwell": C2_FIXED_DWELL,
    }
    if config.learned_priority:
        contract["priority_classes"] = AGENTS
    return contract


def save_c2_checkpoint(
    path: Union[str, Path],
    model: JointCoordinator,
    *,
    step: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> None:
    if not _valid_c2_config(model.config):
        raise ValueError(f"C2 model config mismatch: {model.config}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected_c2_contract(model.config),
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "step": int(step),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extras is not None:
        payload["extras"] = dict(extras)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_c2_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> Tuple[JointCoordinator, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"C2 checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("C2 checkpoint root must be a dict")
    if "model_config" not in payload:
        raise ValueError("C2 checkpoint missing model_config")
    config = CoordinatorConfig(**payload["model_config"])
    if not _valid_c2_config(config):
        raise ValueError(f"C2 model config mismatch: {config}")
    expected = expected_c2_contract(config)
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(f"C2 checkpoint mismatch: missing={missing}, mismatch={mismatch}")
    model = JointCoordinator(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload
