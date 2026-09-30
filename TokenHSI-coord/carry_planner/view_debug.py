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


__all__ = ["rejected_path_vertices", "rejection_reason", "viewer_cross_slots"]
