"""Strict B0 checkpoint I/O, separate from the C1 model contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .schema import AGENTS, MAX_SPEED, MIN_SPEED, PATH_DS, PATH_POINTS, PATH_VERTICES
from .simple_model import SIMPLE_ACTION_DIM, SimpleCoordinatorConfig, SimpleJointCoordinator


SIMPLE_SCHEMA_VERSION = "tokenhsi-coord-b0-v1"


def expected_simple_contract() -> Dict[str, Any]:
    return {
        "schema_version": SIMPLE_SCHEMA_VERSION,
        "model_kind": "simple_joint_mlp",
        "agents": AGENTS,
        "action_dim": SIMPLE_ACTION_DIM,
        "path_points": PATH_POINTS,
        "path_ds": PATH_DS,
        "path_vertices": PATH_VERTICES,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
    }


def save_simple_checkpoint(
    path: Union[str, Path],
    model: SimpleJointCoordinator,
    *,
    step: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected_simple_contract(),
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


def load_simple_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> Tuple[SimpleJointCoordinator, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"simple coordinator checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    expected = expected_simple_contract()
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(f"simple checkpoint mismatch: missing={missing}, mismatch={mismatch}")
    model = SimpleJointCoordinator(SimpleCoordinatorConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload
