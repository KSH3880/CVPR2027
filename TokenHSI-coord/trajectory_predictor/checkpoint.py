"""Strict, self-describing checkpoint I/O for the planner only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from tokenhsi.utils import steer_path as sp

from .model import JointTrajectoryPredictor, ModelConfig
from .schema import (
    AGENTS,
    CANDIDATES,
    COARSE_POINTS,
    HORIZON_SECONDS,
    SCHEMA_VERSION,
    SPEED_VALUES,
    STEER_POINTS,
)


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_contract(dataset_hash: Optional[str] = None) -> Dict[str, Any]:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "agents": AGENTS,
        "coarse_points": COARSE_POINTS,
        "path_ds": sp.DS,
        "path_vertices": sp.V,
        "steer_points": STEER_POINTS,
        "horizon_seconds": HORIZON_SECONDS,
        "speed_values": list(SPEED_VALUES),
        "candidate_k": CANDIDATES,
    }
    if dataset_hash is not None:
        contract["dataset_hash"] = dataset_hash
    return contract


def _validate_contract(payload: Mapping[str, Any]) -> None:
    expected = expected_contract()
    missing = sorted(set(expected) - set(payload))
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if missing or mismatch:
        raise ValueError(f"planner checkpoint contract mismatch: missing={missing}, mismatch={mismatch}")
    dataset_hash = payload.get("dataset_hash")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError("planner checkpoint requires a 64-character training dataset hash")


def save_checkpoint(
    path: Union[str, Path],
    model: JointTrajectoryPredictor,
    dataset_hash: str,
    *,
    epoch: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    if len(dataset_hash) != 64:
        raise ValueError("dataset_hash must be a SHA-256 hex digest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected_contract(dataset_hash),
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "input_normalizer": model.normalizer.payload(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Union[str, Path], map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[JointTrajectoryPredictor, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"planner checkpoint not found: {path}")
    payload = _torch_load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("planner checkpoint root must be a dict")
    _validate_contract(payload)
    if "model_config" not in payload or "model_state" not in payload or "input_normalizer" not in payload:
        raise ValueError("planner checkpoint is missing model config/state/normalizer")
    allowed_config = set(ModelConfig.__dataclass_fields__)
    extra_config = sorted(set(payload["model_config"]) - allowed_config)
    if extra_config:
        raise ValueError(f"unknown planner model config keys: {extra_config}")
    model = JointTrajectoryPredictor(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    # The normalizer also lives in state_dict, but the separate payload is part
    # of the public checkpoint contract. Validate that the two copies agree.
    saved_norm = payload["input_normalizer"]
    for key in ("mean", "std"):
        current = getattr(model.normalizer, key).detach().cpu()
        if key not in saved_norm or not torch.equal(current, saved_norm[key].cpu()):
            raise ValueError(f"checkpoint normalizer {key} disagrees with model_state")
    model.eval()
    return model, payload


def contract_json(payload: Mapping[str, Any]) -> str:
    """Human-readable contract subset for CLI diagnostics."""
    keys = tuple(expected_contract()) + ("dataset_hash", "model_config")
    return json.dumps({key: payload[key] for key in keys}, indent=2, sort_keys=True)
