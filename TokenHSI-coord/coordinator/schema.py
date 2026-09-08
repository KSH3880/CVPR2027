"""Stable state and checkpoint contract for the C1 coordinator.

The public input deliberately consists only of refreshed simulator state and
final goals.  GT trajectories and scenario labels are not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch


SCHEMA_VERSION = "tokenhsi-coord-c1-v1"
AGENTS = 2
CANDIDATES = 4
PATH_POINTS = 33
WAYPOINT_RESIDUAL_POINTS = PATH_POINTS - 3  # all points except root / box / goal
ACCEL_KNOTS = 8
STEER_POINTS = 6
PATH_DS = 0.1
PATH_VERTICES = 320
STEER_HORIZON_SECONDS = 1.6
MIN_SPEED = 0.375
MAX_SPEED = 1.5
MAX_ACCEL = 0.75

STATE_KEYS = (
    "root_xy",
    "heading",
    "root_vel_xy",
    "box_xyz",
    "box_heading",
    "box_vel_xy",
    "box_size_xy",
    "goal_xy",
    "held",
    "phase",
)


@dataclass
class CoordinatorState:
    root_xy: torch.Tensor       # [B, 2, 2]
    heading: torch.Tensor       # [B, 2]
    root_vel_xy: torch.Tensor   # [B, 2, 2]
    box_xyz: torch.Tensor       # [B, 2, 3]
    box_heading: torch.Tensor   # [B, 2]
    box_vel_xy: torch.Tensor    # [B, 2, 2]
    box_size_xy: torch.Tensor   # [B, 2, 2]
    goal_xy: torch.Tensor       # [B, 2, 2]
    held: torch.Tensor          # [B, 2], 0/1 from actual height gate
    phase: torch.Tensor         # [B, 2], approach/wait/carry/place = 0/1/2/3

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "CoordinatorState":
        missing = [key for key in STATE_KEYS if key not in value]
        extra = sorted(set(value) - set(STATE_KEYS))
        if missing or extra:
            raise ValueError(f"coordinator state keys mismatch: missing={missing}, extra={extra}")
        state = cls(**{key: value[key] for key in STATE_KEYS})
        state.validate()
        return state

    def validate(self) -> None:
        tails = {
            "root_xy": (AGENTS, 2),
            "heading": (AGENTS,),
            "root_vel_xy": (AGENTS, 2),
            "box_xyz": (AGENTS, 3),
            "box_heading": (AGENTS,),
            "box_vel_xy": (AGENTS, 2),
            "box_size_xy": (AGENTS, 2),
            "goal_xy": (AGENTS, 2),
            "held": (AGENTS,),
            "phase": (AGENTS,),
        }
        batch = None
        for key, tail in tails.items():
            value = getattr(self, key)
            if not torch.is_tensor(value):
                raise TypeError(f"{key} must be a tensor")
            if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail:
                raise ValueError(f"{key}: expected [B,{','.join(map(str, tail))}], got {tuple(value.shape)}")
            if batch is None:
                batch = value.shape[0]
            elif value.shape[0] != batch:
                raise ValueError("all coordinator state tensors must share the batch dimension")
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite values")
        if ((self.held < 0) | (self.held > 1)).any():
            raise ValueError("held must be in [0, 1]")
        if ((self.phase < 0) | (self.phase > 3)).any():
            raise ValueError("phase must be in [0, 3]")
        if (self.box_size_xy <= 0).any():
            raise ValueError("box_size_xy must be positive")

    @property
    def batch_size(self) -> int:
        return self.root_xy.shape[0]

    @property
    def device(self) -> torch.device:
        return self.root_xy.device

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {key: getattr(self, key) for key in STATE_KEYS}

    def to(self, *args, **kwargs) -> "CoordinatorState":
        return CoordinatorState(**{key: getattr(self, key).to(*args, **kwargs) for key in STATE_KEYS})

    def clone(self) -> "CoordinatorState":
        return CoordinatorState(**{key: getattr(self, key).clone() for key in STATE_KEYS})

    def index(self, index) -> "CoordinatorState":
        return CoordinatorState(**{key: getattr(self, key)[index] for key in STATE_KEYS})
