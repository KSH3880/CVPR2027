"""Pure helpers for rendering and labelling rejected Carry proposals."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


_VALIDITY_LABELS = (
    ("finite", "finite"),
    ("anchors", "anchors"),
    ("buffer", "buffer"),
    ("speed", "speed"),
    ("curve", "curve"),
)


def viewer_cross_slots(start: int, count: int, device) -> torch.Tensor:
    """Reserve one in four viewer episodes for a timed Cross stress test."""
    if start < 0 or count < 0:
        raise ValueError("start and count must be nonnegative")
    return (torch.arange(count, device=device) + start) % 4 == 0


def cap_box_approach_distance(root_xy: torch.Tensor, box_xy: torch.Tensor,
                              max_distance: float) -> torch.Tensor:
    """Limit a walk-start box's XY distance while preserving its direction."""
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    delta = box_xy - root_xy
    distance = delta.norm(dim=-1, keepdim=True)
    scale = (max_distance / distance.clamp(min=1e-7)).clamp(max=1.0)
    return root_xy + scale * delta


def viewer_walk_box_shift(reset_rows, root_rows: torch.Tensor,
                          box_by_agent: torch.Tensor, num_agents: int,
                          max_distance: float):
    """Return selected agent rows and XY shifts, or None before ref init."""
    walk_rows = (reset_rows or {}).get("carry", {}).get("loco_carry")
    if walk_rows is None or len(walk_rows) == 0:
        return None
    env = torch.div(walk_rows, num_agents, rounding_mode="floor")
    agent = walk_rows % num_agents
    old_xy = box_by_agent[env, agent, :2]
    new_xy = cap_box_approach_distance(
        root_rows[walk_rows, :2], old_xy, max_distance,
    )
    return env, agent, new_xy - old_xy


def rejected_path_vertices(path_xy, height: float = 0.08) -> np.ndarray:
    """Convert [agents, points, 2] paths to Isaac Gym line vertices."""
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 3 or path.shape[-1] != 2 or path.shape[-2] < 2:
        raise ValueError("path_xy must have shape [agents,points>=2,2]")
    z = np.full(path.shape[:-1] + (1,), float(height), dtype=np.float32)
    xyz = np.concatenate((path, z), axis=-1)
    return np.concatenate((xyz[:, :-1], xyz[:, 1:]), axis=-1)


def rejection_reason(
    diagnostics: Mapping[str, torch.Tensor], index: int,
) -> str:
    """Return the hard-validity predicates that rejected one proposal."""
    failed = [
        label for key, label in _VALIDITY_LABELS
        if key in diagnostics and not bool(diagnostics[key][index])
    ]
    return "+".join(failed) if failed else "unknown"


__all__ = [
    "cap_box_approach_distance", "rejected_path_vertices", "rejection_reason",
    "viewer_cross_slots", "viewer_walk_box_shift",
]
