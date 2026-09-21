"""Strict, atomic checkpoint contract for coordinator_v2."""

from __future__ import annotations

import os
from pathlib import Path
import torch

from .core import PlannerConfig, WorldModelPlanner

SCHEMA_VERSION = "tokenhsi-wm-mppi-v1"


def save_checkpoint(path, planner, *, step=0, optimizer=None, extras=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "config": planner.config.as_dict(),
               "model_state": planner.state_dict(), "step": int(step), "extras": dict(extras or {})}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    payload = torch.load(path, map_location=device)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("coordinator_v2 checkpoint schema mismatch")
    allowed = set(PlannerConfig.__dataclass_fields__)
    extra = sorted(set(payload.get("config", {})) - allowed)
    if extra:
        raise ValueError(f"unknown coordinator_v2 config keys: {extra}")
    planner = WorldModelPlanner(PlannerConfig(**payload["config"])).to(device)
    planner.load_state_dict(payload["model_state"], strict=True)
    planner.eval()
    return planner, payload
