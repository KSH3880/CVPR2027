"""Strict, atomic checkpoint I/O for the isolated stack planner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .model import StackPlannerConfig, StackTrajectoryPlanner
from .schema import (
    ACCEL_KNOTS,
    AGENTS,
    MAX_ACCEL,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    STACK_SCHEMA_VERSION,
    STEER_HORIZON_SECONDS,
    STEER_POINTS,
)


def expected_contract(config: StackPlannerConfig) -> Dict[str, Any]:
    return {
        "schema_version": STACK_SCHEMA_VERSION,
        "model_kind": "stack_transformer",
        "agents": AGENTS,
        "candidate_k": config.candidates,
        "path_points": PATH_POINTS,
        "retreat_path_points": PATH_POINTS,
        "retreat_distance": config.retreat_distance,
        "accel_knots": ACCEL_KNOTS,
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "steer_points": STEER_POINTS,
        "steer_horizon_seconds": STEER_HORIZON_SECONDS,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
        "max_accel": MAX_ACCEL,
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
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stack planner checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("stack planner checkpoint root must be a dict")
    for key in ("model_config", "model_state"):
        if key not in payload:
            raise ValueError(f"stack planner checkpoint missing {key}")
    allowed = set(StackPlannerConfig.__dataclass_fields__)
    extra = sorted(set(payload["model_config"]) - allowed)
    if extra:
        raise ValueError(f"unknown stack planner config keys: {extra}")
    config = StackPlannerConfig(**payload["model_config"])
    # Early v1 checkpoints already contain the retreat heads and the
    # retreat_distance model config, but predate these two redundant contract
    # fields.  Reconstruct only those derivable fields; every stored value is
    # still checked strictly below.
    if payload.get("schema_version") == STACK_SCHEMA_VERSION:
        payload.setdefault("retreat_path_points", PATH_POINTS)
        payload.setdefault("retreat_distance", config.retreat_distance)
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
