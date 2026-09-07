"""Strict and atomic coordinator-only checkpoint I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .model import CoordinatorConfig, JointCoordinator
from .schema import (
    ACCEL_KNOTS,
    AGENTS,
    CANDIDATES,
    MAX_ACCEL,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    SCHEMA_VERSION,
    STEER_HORIZON_SECONDS,
    STEER_POINTS,
)


def expected_contract() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "agents": AGENTS,
        "candidate_k": CANDIDATES,
        "path_points": PATH_POINTS,
        "accel_knots": ACCEL_KNOTS,
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "steer_points": STEER_POINTS,
        "steer_horizon_seconds": STEER_HORIZON_SECONDS,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
        "max_accel": MAX_ACCEL,
    }


def _validate_contract(payload: Mapping[str, Any]) -> None:
    expected = expected_contract()
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(f"coordinator checkpoint contract mismatch: missing={missing}, mismatch={mismatch}")


def save_checkpoint(
    path: Union[str, Path],
    model: JointCoordinator,
    *,
    step: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected_contract(),
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


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> Tuple[JointCoordinator, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"coordinator checkpoint not found: {path}")
    payload = _torch_load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("coordinator checkpoint root must be a dict")
    _validate_contract(payload)
    for key in ("model_config", "model_state"):
        if key not in payload:
            raise ValueError(f"coordinator checkpoint missing {key}")
    allowed = set(CoordinatorConfig.__dataclass_fields__)
    extra = sorted(set(payload["model_config"]) - allowed)
    if extra:
        raise ValueError(f"unknown coordinator model config keys: {extra}")
    model = JointCoordinator(CoordinatorConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload
