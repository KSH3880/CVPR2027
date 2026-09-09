"""Coordinate transforms and path sampling shared by training and runtime."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from tokenhsi.utils import steer_path as sp

from .schema import AGENTS, COARSE_POINTS, PlannerState


POSITION_SCALE = 10.0
HEIGHT_SCALE = 2.0
VELOCITY_SCALE = 3.0
TOKEN_DIM = 8
TOKENS = AGENTS * 3


def rotate_xy(value: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate the final xy dimension by a broadcastable angle in radians."""
    c, s = torch.cos(angle), torch.sin(angle)
    while c.ndim < value.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    x, y = value[..., 0], value[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def shared_frame(state: PlannerState) -> Tuple[torch.Tensor, torch.Tensor]:
    """World -> shared frame: root midpoint origin and A0 heading orientation."""
    center = state.root_xy.mean(dim=1)
    angle = state.heading[:, 0]
    return center, angle


def world_to_shared(points: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    while center.ndim < points.ndim:
        center = center.unsqueeze(-2)
    return rotate_xy(points - center, -angle)


def shared_to_world(points: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    out = rotate_xy(points, angle)
    while center.ndim < out.ndim:
        center = center.unsqueeze(-2)
    return out + center


def state_to_tokens(state: PlannerState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create six entity tokens: (root, box, goal) for stable A/B slots."""
    state.validate()
    center, angle = shared_frame(state)
    root = world_to_shared(state.root_xy, center, angle)
    box = world_to_shared(state.box_xyz[..., :2], center, angle)
    goal = world_to_shared(state.goal_xy, center, angle)
    root_vel = rotate_xy(state.root_vel_xy, -angle)
    box_vel = rotate_xy(state.box_vel_xy, -angle)
    rel_heading = state.heading - angle[:, None]

    b = state.batch_size
    tokens = torch.zeros(b, AGENTS, 3, TOKEN_DIM, device=state.device, dtype=state.root_xy.dtype)
    tokens[:, :, 0, 0:2] = root / POSITION_SCALE
    tokens[:, :, 0, 2] = torch.sin(rel_heading)
    tokens[:, :, 0, 3] = torch.cos(rel_heading)
    tokens[:, :, 0, 4:6] = root_vel / VELOCITY_SCALE
    tokens[:, :, 0, 6] = state.phase

    tokens[:, :, 1, 0:2] = box / POSITION_SCALE
    tokens[:, :, 1, 2] = state.box_xyz[..., 2] / HEIGHT_SCALE
    tokens[:, :, 1, 4:6] = box_vel / VELOCITY_SCALE
    tokens[:, :, 1, 6] = state.phase

    tokens[:, :, 2, 0:2] = goal / POSITION_SCALE
    tokens[:, :, 2, 6] = state.phase

    # Agent-major flatten: A.root,A.box,A.goal,B.root,B.box,B.goal.
    tokens = tokens.reshape(b, TOKENS, TOKEN_DIM)
    frame = {"center": center, "angle": angle, "root": root, "box": box, "goal": goal}
    return tokens, frame


def baseline_path_local(frame: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Piecewise-linear root->box->goal baseline with exact 0/16/32 anchors."""
    root, box, goal = frame["root"], frame["box"], frame["goal"]
    dtype, device = root.dtype, root.device
    t = torch.linspace(0.0, 1.0, 17, device=device, dtype=dtype)[None, None, :, None]
    first = root[:, :, None] + t * (box - root)[:, :, None]
    second = box[:, :, None] + t * (goal - box)[:, :, None]
    return torch.cat((first, second[:, :, 1:]), dim=2)


def path_to_shared(path: torch.Tensor, state: PlannerState) -> torch.Tensor:
    center, angle = shared_frame(state)
    return world_to_shared(path, center, angle)


def path_to_world(path: torch.Tensor, state: PlannerState) -> torch.Tensor:
    center, angle = shared_frame(state)
    return shared_to_world(path, center, angle)


def resample_coarse(path: torch.Tensor, with_end: bool = False):
    """[B,2,33,2] -> inherited [B,2,320,2] steer_path representation."""
    if path.ndim != 4 or tuple(path.shape[1:3]) != (AGENTS, COARSE_POINTS):
        raise ValueError(f"expected [B,2,33,2], got {tuple(path.shape)}")
    b = path.shape[0]
    result = sp.resample(path.reshape(b * AGENTS, COARSE_POINTS, 2), with_end=with_end)
    if with_end:
        dense, end_s = result
        return dense.reshape(b, AGENTS, sp.V, 2), end_s.reshape(b, AGENTS)
    return result.reshape(b, AGENTS, sp.V, 2)


def sample_coarse_from_dense(
    dense: torch.Tensor,
    s_box: torch.Tensor,
    end_s: torch.Tensor,
    box_xy: torch.Tensor,
    goal_xy: torch.Tensor,
) -> torch.Tensor:
    """Sample 17 points on each leg and re-assert the three hard anchors."""
    n = dense.shape[0]
    t = torch.linspace(0.0, 1.0, 17, device=dense.device, dtype=dense.dtype)[None]
    first_s = s_box[:, None] * t
    second_s = s_box[:, None] + (end_s - s_box)[:, None] * t
    arc = torch.cat((first_s, second_s[:, 1:]), dim=1)
    q = (arc / sp.DS).clamp(0, sp.V - 2)
    lo = q.floor().long()
    frac = (q - lo).unsqueeze(-1)
    ar = torch.arange(n, device=dense.device)[:, None]
    points = dense[ar, lo] + frac * (dense[ar, lo + 1] - dense[ar, lo])
    points[:, 0] = dense[:, 0]
    points[:, 16] = box_xy
    points[:, 32] = goal_xy
    return points


def augment_state_and_path(
    state: PlannerState,
    path: torch.Tensor,
    seed: Optional[int] = None,
) -> Tuple[PlannerState, torch.Tensor]:
    """Apply one random rigid transform per scene without changing labels."""
    g = torch.Generator(device="cpu")
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, ()).item())
    g.manual_seed(int(seed))
    b = state.batch_size
    angle = (torch.rand(b, generator=g) * 2.0 - 1.0).to(state.root_xy) * math.pi
    shift = (torch.rand(b, 2, generator=g) * 20.0 - 10.0).to(state.root_xy)

    def pos(x):
        y = rotate_xy(x, angle)
        s = shift
        while s.ndim < y.ndim:
            s = s.unsqueeze(-2)
        return y + s

    result = state.clone()
    result.root_xy = pos(result.root_xy)
    result.box_xyz[..., :2] = pos(result.box_xyz[..., :2])
    result.goal_xy = pos(result.goal_xy)
    result.root_vel_xy = rotate_xy(result.root_vel_xy, angle)
    result.box_vel_xy = rotate_xy(result.box_vel_xy, angle)
    result.heading = (result.heading + angle[:, None] + math.pi) % (2 * math.pi) - math.pi
    return result, pos(path)
