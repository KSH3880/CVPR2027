"""Stable V1 tensor/checkpoint contract.

The package deliberately stores state as named tensors rather than one opaque
vector.  That makes accidental use of a semantic role, scenario id, or joint
state impossible to hide in a changing feature order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch


SCHEMA_VERSION = "masteer-joint-trajectory-v1"
AGENTS = 2
COARSE_POINTS = 33
STEER_POINTS = 6
CANDIDATES = 1
HORIZON_SECONDS = 1.6
SPEED_VALUES = (0.375, 0.75, 1.125, 1.5)

STATE_KEYS = (
    "root_xy",
    "heading",
    "root_vel_xy",
    "box_xyz",
    "box_vel_xy",
    "goal_xy",
    "phase",
)


@dataclass
class PlannerState:
    root_xy: torch.Tensor       # [B, 2, 2]
    heading: torch.Tensor       # [B, 2], yaw radians
    root_vel_xy: torch.Tensor   # [B, 2, 2]
    box_xyz: torch.Tensor       # [B, 2, 3]
    box_vel_xy: torch.Tensor    # [B, 2, 2]
    goal_xy: torch.Tensor       # [B, 2, 2]
    phase: torch.Tensor         # [B, 2], 0=approach, 1=carry

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "PlannerState":
        missing = [key for key in STATE_KEYS if key not in value]
        extra = sorted(set(value) - set(STATE_KEYS))
        if missing or extra:
            raise ValueError(f"planner state keys mismatch: missing={missing}, extra={extra}")
        state = cls(**{key: value[key] for key in STATE_KEYS})
        state.validate()
        return state

    def validate(self) -> None:
        expected_tail = {
            "root_xy": (AGENTS, 2),
            "heading": (AGENTS,),
            "root_vel_xy": (AGENTS, 2),
            "box_xyz": (AGENTS, 3),
            "box_vel_xy": (AGENTS, 2),
            "goal_xy": (AGENTS, 2),
            "phase": (AGENTS,),
        }
        batch = None
        for key, tail in expected_tail.items():
            tensor = getattr(self, key)
            if not torch.is_tensor(tensor):
                raise TypeError(f"{key} must be a tensor")
            if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
                raise ValueError(f"{key}: expected [B,{','.join(map(str, tail))}], got {tuple(tensor.shape)}")
            batch = tensor.shape[0] if batch is None else batch
            if tensor.shape[0] != batch:
                raise ValueError("all planner state tensors must share the batch dimension")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key} contains non-finite values")
        if ((self.phase < 0) | (self.phase > 1)).any():
            raise ValueError("phase must be in [0, 1]")

    @property
    def batch_size(self) -> int:
        return self.root_xy.shape[0]

    @property
    def device(self) -> torch.device:
        return self.root_xy.device

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {key: getattr(self, key) for key in STATE_KEYS}

    def to(self, *args, **kwargs) -> "PlannerState":
        return PlannerState(**{key: getattr(self, key).to(*args, **kwargs) for key in STATE_KEYS})

    def clone(self) -> "PlannerState":
        return PlannerState(**{key: getattr(self, key).clone() for key in STATE_KEYS})

    def index(self, index) -> "PlannerState":
        return PlannerState(**{key: getattr(self, key)[index] for key in STATE_KEYS})


def state_from_batch(batch: Mapping[str, torch.Tensor]) -> PlannerState:
    """Read a collated dataset batch without accepting unrelated features."""
    return PlannerState.from_mapping({key: batch[key] for key in STATE_KEYS})
