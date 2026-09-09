"""CPU scene synthesis, serialization, and PyTorch Dataset wrappers."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple, Union

import torch
from torch.utils.data import Dataset

from .geometry import augment_state_and_path, rotate_xy
from .oracle import solve_oracle
from .schema import AGENTS, SCHEMA_VERSION, STATE_KEYS, PlannerState


DEFAULT_SPLITS = {"train": 200_000, "val": 20_000, "test": 20_000}
SPLIT_SEEDS = {"train": 1_000_003, "val": 2_000_003, "test": 3_000_001}


def _pairwise_clear(points: torch.Tensor, minimum: float = 1.0) -> torch.Tensor:
    distance = torch.cdist(points, points)
    eye = torch.eye(points.shape[1], dtype=torch.bool)[None]
    distance = distance.masked_fill(eye, float("inf"))
    return distance.amin(dim=(1, 2)) >= minimum


def _sample_free(count: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    roots, boxes, goals = [], [], []
    remaining = count
    while remaining:
        n = max(remaining * 3, 64)
        values = torch.rand(n, 6, 2, generator=generator) * 9.0 - 4.5
        valid = _pairwise_clear(values)
        values = values[valid][:remaining]
        if len(values) == 0:
            continue
        roots.append(values[:, 0:2])
        boxes.append(values[:, 2:4])
        goals.append(values[:, 4:6])
        remaining -= len(values)
    return torch.cat(roots), torch.cat(boxes), torch.cat(goals)


def _sample_cross(count: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = torch.tensor([[[-4.0, -1.0], [1.0, -4.0]]]).expand(count, -1, -1).clone()
    box = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]]).expand(count, -1, -1).clone()
    goal = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]]).expand(count, -1, -1).clone()
    root += (torch.rand(count, AGENTS, 2, generator=generator) - 0.5) * 0.3
    box += (torch.rand(count, AGENTS, 2, generator=generator) - 0.5) * 0.3
    goal += (torch.rand(count, AGENTS, 2, generator=generator) - 0.5) * 0.3
    angle = (torch.rand(count, generator=generator) * 2.0 - 1.0) * math.pi
    shift = (torch.rand(count, 2, generator=generator) - 0.5) * 1.0
    root = rotate_xy(root, angle) + shift[:, None]
    box = rotate_xy(box, angle) + shift[:, None]
    goal = rotate_xy(goal, angle) + shift[:, None]
    return root, box, goal


def sample_states(count: int, seed: int) -> Tuple[PlannerState, torch.Tensor, torch.Tensor]:
    """50:50 Free/Cross states in the ms18 [-4.5,4.5] workspace."""
    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    is_cross = torch.arange(count) % 2 == 1
    free_n, cross_n = int((~is_cross).sum()), int(is_cross.sum())
    fr, fb, fg = _sample_free(free_n, generator)
    cr, cb, cg = _sample_cross(cross_n, generator)
    root = torch.empty(count, AGENTS, 2)
    box_xy = torch.empty_like(root)
    goal = torch.empty_like(root)
    root[~is_cross], box_xy[~is_cross], goal[~is_cross] = fr, fb, fg
    root[is_cross], box_xy[is_cross], goal[is_cross] = cr, cb, cg

    # Phase is observable state, not a role. Carry examples collapse the unused
    # first leg so replans after pickup are represented during pretraining.
    phase = (torch.rand(count, AGENTS, generator=generator) < 0.35).float()
    carry_noise = (torch.rand(count, AGENTS, 2, generator=generator) - 0.5) * 0.12
    root = torch.where(phase[..., None].bool(), box_xy + carry_noise, root)
    direction = torch.where(phase[..., None].bool(), goal - root, box_xy - root)
    heading = torch.atan2(direction[..., 1], direction[..., 0])
    heading += (torch.rand(count, AGENTS, generator=generator) - 0.5) * 0.5
    speed = torch.rand(count, AGENTS, 1, generator=generator) * 1.5
    root_vel = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * speed
    root_vel += torch.randn(count, AGENTS, 2, generator=generator) * 0.08
    box_vel = torch.where(
        phase[..., None].bool(), root_vel + torch.randn(count, AGENTS, 2, generator=generator) * 0.05,
        torch.zeros_like(root_vel),
    )
    ground_z = 0.25 + torch.rand(count, AGENTS, generator=generator) * 0.25
    carry_z = 0.75 + torch.rand(count, AGENTS, generator=generator) * 0.45
    box_z = torch.where(phase.bool(), carry_z, ground_z)
    state = PlannerState(
        root_xy=root,
        heading=heading,
        root_vel_xy=root_vel,
        box_xyz=torch.cat((box_xy, box_z[..., None]), dim=-1),
        box_vel_xy=box_vel,
        goal_xy=goal,
        phase=phase,
    )
    scene_seed = torch.arange(seed, seed + count, dtype=torch.int64)
    return state, is_cross.to(torch.uint8), scene_seed


def generate_split(count: int, seed: int, batch_size: int = 256) -> Dict[str, object]:
    """Generate one serializable split without importing Isaac Gym."""
    states, scenario, scene_seed = sample_states(count, seed)
    outputs = {key: [] for key in ("coarse_path", "speed_class", "oracle_valid", "min_clearance", "makespan")}
    for lo in range(0, count, batch_size):
        hi = min(lo + batch_size, count)
        result = solve_oracle(states.index(slice(lo, hi)))
        for key, value in result.as_dict().items():
            outputs[key].append(value.cpu())
    return {
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "seed": seed,
        "state": {key: value.cpu() for key, value in states.as_dict().items()},
        "scenario": scenario,
        "scene_seed": scene_seed,
        **{key: torch.cat(value, dim=0) for key, value in outputs.items()},
    }


def save_split(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_split(path: Union[str, Path]) -> Dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"dataset schema mismatch: {payload.get('schema_version')}")
    PlannerState.from_mapping(payload["state"])
    count = int(payload["count"])
    for key in ("coarse_path", "speed_class", "oracle_valid", "scenario", "scene_seed"):
        if key not in payload or len(payload[key]) != count:
            raise ValueError(f"dataset field {key} has the wrong length")
    return payload


class TrajectoryDataset(Dataset):
    def __init__(self, path: Union[str, Path], *, augment: bool = False, valid_only: bool = False):
        self.payload = load_split(path)
        self.state = PlannerState.from_mapping(self.payload["state"])
        self.augment = augment
        if valid_only:
            self.indices = self.payload["oracle_valid"].nonzero(as_tuple=False).squeeze(-1)
        else:
            self.indices = torch.arange(self.state.batch_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        index = int(self.indices[item])
        state = self.state.index(slice(index, index + 1))
        path = self.payload["coarse_path"][index:index + 1]
        if self.augment:
            state, path = augment_state_and_path(state, path)
        result = {key: value[0] for key, value in state.as_dict().items()}
        result.update({
            "coarse_path": path[0],
            "speed_class": self.payload["speed_class"][index],
            "oracle_valid": self.payload["oracle_valid"][index],
            "scenario": self.payload["scenario"][index],
        })
        return result
