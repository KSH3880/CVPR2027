"""Strict, atomic checkpoint I/O for the isolated stack planner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .model import StackPlannerConfig, StackTrajectoryPlanner
from .schema import (
    AGENTS,
    MAX_ACCEL,
    PATH_DS,
    PATH_VERTICES,
    STACK_SCHEMA_VERSION,
    STACK_PATH_POINTS,
    STEER_HORIZON_SECONDS,
    STEER_POINTS,
)

LEGACY_VIEW_SCHEMA_V13 = "tokenhsi-stack-planner-v13"
LEGACY_V13_CONFIG_FIELDS = {
    "token_dim", "d_model", "nhead", "encoder_layers", "feedforward",
    "dropout", "candidates", "delta_scale", "history_steps",
}


def expected_contract(config: StackPlannerConfig) -> Dict[str, Any]:
    return {
        "schema_version": STACK_SCHEMA_VERSION,
        "model_kind": "stack_scene_token_multihead",
        "agents": AGENTS,
        "candidate_k": config.candidates,
        "path_points": STACK_PATH_POINTS,
        "path_only": False,
        "pointwise_speed_profile": True,
        "runtime_acceleration_limit": MAX_ACCEL,
        "full_candidate_rollout": True,
        "previous_trajectory_correction": True,
        "bounded_absolute_path_target": True,
        "path_update_alpha": config.path_update_alpha,
        "retreat_delta_scale": config.retreat_delta_scale,
        "planner_task": "retreat_only" if config.retreat_only else "full_stack",
        "fixed_origin_reference": True,
        "future_action_mask": True,
        "projected_executor_resume": True,
        "joint_a2_preplan": True,
        "deferred_a2_carry_goal": True,
        "smooth_path_regularization": True,
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "steer_points": STEER_POINTS,
        "steer_horizon_seconds": STEER_HORIZON_SECONDS,
    }


def save_stack_checkpoint(
    path: Union[str, Path],
    model: StackTrajectoryPlanner,
    *,
    step: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected_contract(model.config),
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


def load_stack_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[StackTrajectoryPlanner, Dict[str, Any]]:
    payload = _read_checkpoint_payload(path, device)
    return _load_current_payload(payload, device)


def _read_checkpoint_payload(
    path: Union[str, Path], device: Union[str, torch.device],
) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stack planner checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("stack planner checkpoint root must be a dict")
    return payload


def _load_current_payload(
    payload: Dict[str, Any], device: Union[str, torch.device],
) -> Tuple[StackTrajectoryPlanner, Dict[str, Any]]:
    for key in ("model_config", "model_state"):
        if key not in payload:
            raise ValueError(f"stack planner checkpoint missing {key}")
    allowed = set(StackPlannerConfig.__dataclass_fields__)
    extra = sorted(set(payload["model_config"]) - allowed)
    if extra:
        raise ValueError(f"unknown stack planner config keys: {extra}")
    config = StackPlannerConfig(**payload["model_config"])
    expected = expected_contract(config)
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(
            f"stack planner checkpoint mismatch: missing={missing}, mismatch={mismatch}"
        )
    model = StackTrajectoryPlanner(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def _expected_v13_contract(config: StackPlannerConfig) -> Dict[str, Any]:
    """Frozen V13 contract used only to reject ambiguous legacy files."""
    return {
        "schema_version": LEGACY_VIEW_SCHEMA_V13,
        "model_kind": "stack_scene_token_multihead",
        "agents": AGENTS,
        "candidate_k": config.candidates,
        "path_points": STACK_PATH_POINTS,
        "path_only": False,
        "pointwise_speed_profile": True,
        "runtime_acceleration_limit": MAX_ACCEL,
        "full_candidate_rollout": True,
        "previous_trajectory_correction": True,
        "fixed_origin_reference": True,
        "future_action_mask": True,
        "projected_executor_resume": True,
        "joint_a2_preplan": True,
        "deferred_a2_carry_goal": True,
        "smooth_path_regularization": True,
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "steer_points": STEER_POINTS,
        "steer_horizon_seconds": STEER_HORIZON_SECONDS,
    }


def load_stack_checkpoint_for_view(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[StackTrajectoryPlanner, Dict[str, Any]]:
    """Load current checkpoints or faithfully decode V13 for viewing only.

    Training and resume continue to use :func:`load_stack_checkpoint`, which
    deliberately rejects all old schemas.
    """
    payload = _read_checkpoint_payload(path, device)
    if payload.get("schema_version") != LEGACY_VIEW_SCHEMA_V13:
        return _load_current_payload(payload, device)
    for key in ("model_config", "model_state"):
        if key not in payload:
            raise ValueError(f"stack planner checkpoint missing {key}")
    config_values = payload["model_config"]
    extra = sorted(set(config_values) - LEGACY_V13_CONFIG_FIELDS)
    missing_config = sorted(LEGACY_V13_CONFIG_FIELDS - set(config_values))
    if extra or missing_config:
        raise ValueError(
            "V13 viewer config mismatch: "
            f"missing={missing_config}, unknown={extra}"
        )
    # These V15-only values are inert in the V13 implementation below.  They
    # merely let the shared validated config describe the unchanged network.
    config = StackPlannerConfig(
        **config_values,
        retreat_delta_scale=config_values["delta_scale"],
        path_update_alpha=1.0,
        retreat_only=False,
    )
    expected = _expected_v13_contract(config)
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(
            f"V13 viewer checkpoint mismatch: missing={missing}, "
            f"mismatch={mismatch}"
        )
    from .legacy_v13 import StackTrajectoryPlannerV13
    model = StackTrajectoryPlannerV13(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    payload = dict(payload)
    payload["viewer_legacy_decoder"] = True
    return model, payload
