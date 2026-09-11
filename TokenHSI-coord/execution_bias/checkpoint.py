"""Strict checkpoint I/O for deterministic execution-bias models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from .model import BIAS_SCHEMA_VERSION, ExecutionBiasConfig, ExecutionBiasMLP


def save_bias_checkpoint(
    path: str | Path,
    model: ExecutionBiasMLP,
    *,
    step: int = 0,
    metrics: Mapping[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BIAS_SCHEMA_VERSION,
        "model_kind": "deterministic_execution_bias_mlp",
        "model_config": model.config.as_dict(),
        "model_state": model.state_dict(),
        "step": int(step),
        "metrics": dict(metrics or {}),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bias_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[ExecutionBiasMLP, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"execution-bias checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema_version", "model_kind", "model_config", "model_state",
        "step", "metrics",
    }
    if set(payload) != expected:
        raise ValueError(
            f"execution-bias checkpoint keys mismatch: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    if payload["schema_version"] != BIAS_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {payload['schema_version']}")
    if payload["model_kind"] != "deterministic_execution_bias_mlp":
        raise ValueError(f"unsupported model kind: {payload['model_kind']}")
    model = ExecutionBiasMLP(ExecutionBiasConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload
