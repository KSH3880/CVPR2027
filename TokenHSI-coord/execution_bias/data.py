"""Dataset contract and nominal-time alignment for executor rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def nominal_times(plan_xyv: np.ndarray, minimum_speed: float = 0.05) -> np.ndarray:
    """Compute waypoint timestamps for [...,A,P,3] spatial path/speed plans."""
    plan = np.asarray(plan_xyv)
    if plan.shape[-1] != 3 or plan.shape[-2] < 2:
        raise ValueError("plan_xyv must end in [P,3] with P >= 2")
    distance = np.linalg.norm(np.diff(plan[..., :2], axis=-2), axis=-1)
    speed = np.maximum(plan[..., :-1, 2], minimum_speed)
    duration = distance / speed
    zero = np.zeros(duration.shape[:-1] + (1,), dtype=duration.dtype)
    return np.concatenate((zero, np.cumsum(duration, axis=-1)), axis=-1)


def align_actual_to_plan(
    plan_xyv: np.ndarray,
    actual_time: np.ndarray,
    actual_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one rollout onto each agent's nominal plan timestamps.

    Args:
        plan_xyv: [A,P,3]
        actual_time: [T], seconds from plan installation
        actual_xy: [T,A,2]
    Returns:
        error_xy: [A,P,2]
        valid: [A,P], false beyond the recorded rollout horizon
    """
    plan = np.asarray(plan_xyv, dtype=np.float64)
    time = np.asarray(actual_time, dtype=np.float64)
    actual = np.asarray(actual_xy, dtype=np.float64)
    if plan.ndim != 3 or plan.shape[-1] != 3:
        raise ValueError("plan_xyv must be [A,P,3]")
    if time.ndim != 1 or actual.shape != (time.size, plan.shape[0], 2):
        raise ValueError("actual_time/actual_xy must be [T] and [T,A,2]")
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("actual_time must be strictly increasing with at least two samples")
    target_time = nominal_times(plan)
    aligned = np.empty(plan.shape[:-1] + (2,), dtype=np.float64)
    valid = target_time <= time[-1]
    for agent in range(plan.shape[0]):
        for axis in range(2):
            aligned[agent, :, axis] = np.interp(
                target_time[agent], time, actual[:, agent, axis]
            )
    error = aligned - plan[..., :2]
    error[~valid] = 0.0
    return error.astype(np.float32), valid


class ExecutionBiasDataset(Dataset):
    """Read immutable ``plan_xyv/error_xy/valid`` arrays from one NPZ."""

    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {"plan_xyv", "error_xy", "valid"}
            if set(data.files) != required:
                raise ValueError(
                    f"dataset keys must be exactly {sorted(required)}, got {sorted(data.files)}"
                )
            self.plan = torch.from_numpy(data["plan_xyv"].astype(np.float32))
            self.error = torch.from_numpy(data["error_xy"].astype(np.float32))
            self.valid = torch.from_numpy(data["valid"].astype(bool))
        if self.plan.ndim != 4 or self.plan.shape[-1] != 3:
            raise ValueError("plan_xyv must be [N,A,P,3]")
        if self.error.shape != self.plan.shape[:-1] + (2,):
            raise ValueError("error_xy must be [N,A,P,2]")
        if self.valid.shape != self.plan.shape[:-1]:
            raise ValueError("valid must be [N,A,P]")
        if not torch.isfinite(self.plan).all() or not torch.isfinite(self.error).all():
            raise ValueError("dataset contains non-finite values")

    def __len__(self) -> int:
        return self.plan.shape[0]

    def __getitem__(self, index: int):
        return self.plan[index], self.error[index], self.valid[index]
