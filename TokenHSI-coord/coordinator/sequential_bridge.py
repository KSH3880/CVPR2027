"""Pure-PyTorch adapter for phase-owned sequential-stack execution.

The learned model retains its state-only checkpoint contract. The executor owns
the stack dependency and XYZ placement goals; only active Carry rows get plans.
"""

from pathlib import Path

import torch

from .checkpoint import _torch_load, load_checkpoint
from .c2_checkpoint import load_c2_checkpoint
from .simple_checkpoint import load_simple_checkpoint
from .planner import timed_rollout
from .schema import MAX_SPEED, MIN_SPEED, PATH_DS, PATH_POINTS, PATH_VERTICES


def load_planner(path, device):
    path = Path(path).expanduser().resolve()
    payload = _torch_load(path, "cpu")
    loaders = {
        "tokenhsi-coord-c1-v1": load_checkpoint,
        "tokenhsi-coord-c2-v1": load_c2_checkpoint,
        "tokenhsi-coord-b0-v1": load_simple_checkpoint,
    }
    schema = payload.get("schema_version")
    if schema not in loaders:
        raise ValueError("COORD_CKPT must be a coordinator checkpoint, not a masteer PTH")
    model, payload = loaders[schema](path, device)
    model.eval().requires_grad_(False)
    return model, payload


def carry_rows(phase, rehearsal):
    """Sequential ownership: A1 place/verify/retreat, then A2 place/verify."""
    return torch.stack(((phase == 0) | (phase == 1) | (phase == 2),
                        (phase == 3) | (phase == 4)), dim=-1) | rehearsal[:, None]


def plan_validity(path, speed, state, active):
    """Per-candidate geometry, anchors and buffer checks; parked rows ignored."""
    if path.shape != (*speed.shape, 2) or path.shape[2:] != (2, PATH_POINTS, 2):
        raise ValueError("expected [B,K,2,33,2] path and [B,K,2,33] speed")
    finite = torch.isfinite(path).flatten(start_dim=-2).all(dim=-1) & torch.isfinite(speed).all(dim=-1)
    anchors = ((path[..., 0, :] - state.root_xy[:, None]).norm(dim=-1) < 0.01)
    anchors &= ((path[..., 16, :] - state.box_xyz[:, None, :, :2]).norm(dim=-1) < 0.01)
    anchors &= ((path[..., -1, :] - state.goal_xy[:, None]).norm(dim=-1) < 0.01)
    delta = path[..., 1:, :] - path[..., :-1, :]
    lengths = delta.norm(dim=-1)
    buffer_ok = lengths.sum(dim=-1) < (PATH_VERTICES - 2) * PATH_DS
    product = lengths[..., :-1] * lengths[..., 1:]
    cosine = (delta[..., :-1, :] * delta[..., 1:, :]).sum(dim=-1) / product.clamp(min=1e-8)
    turns = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    turns = torch.where(product > 1e-10, turns, torch.zeros_like(turns))
    turns[..., 14:17] = 0
    turns[..., :16] = torch.where(state.held[:, None, :, None] > 0.5,
                                  torch.zeros_like(turns[..., :16]), turns[..., :16])
    speeds_ok = ((speed >= MIN_SPEED) & (speed <= MAX_SPEED)).all(dim=-1)
    valid = finite & anchors & buffer_ok & speeds_ok & (turns.amax(dim=-1) <= 46)
    return (valid | ~active[:, None]).all(dim=-1)


def select_stack_plan(output, state, active):
    """Safety-first selection with inactive agents/boxes stationary."""
    path, speed = output["path_world"], output["speed"]
    valid = plan_validity(path, speed, state, active)
    path = torch.nan_to_num(path, nan=0.0, posinf=0.0, neginf=0.0)
    speed = torch.nan_to_num(speed, nan=MIN_SPEED).clamp(MIN_SPEED, MAX_SPEED)
    path = torch.where(active[:, None, :, None, None], path,
                       state.root_xy[:, None, :, None].expand_as(path))
    speed = torch.where(active[:, None, :, None], speed, torch.full_like(speed, MIN_SPEED))
    dwell = torch.where(active[:, None], output["pickup_dwell"],
                        torch.zeros_like(output["pickup_dwell"]))
    valid &= torch.isfinite(dwell).all(dim=-1) & (dwell >= 0).all(dim=-1)
    dwell = torch.nan_to_num(dwell, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    future = timed_rollout(path, speed, dwell, state)
    root = torch.where(active[:, None, :, None, None], future["root"],
                       state.root_xy[:, None, :, None])
    box = torch.where(active[:, None, :, None, None], future["box"],
                      state.box_xyz[:, None, :, None, :2])
    hh = (root[:, :, 0] - root[:, :, 1]).norm(dim=-1)
    bb = (box[:, :, 0] - box[:, :, 1]).norm(dim=-1)
    hb01 = (root[:, :, 0] - box[:, :, 1]).norm(dim=-1)
    hb10 = (root[:, :, 1] - box[:, :, 0]).norm(dim=-1)
    radius = 0.5 * state.box_size_xy.norm(dim=-1)
    bb_limit = radius.sum(dim=-1)[:, None, None] + 0.15
    hb01_limit = radius[:, 1, None, None] + 0.35 + 0.15
    hb10_limit = radius[:, 0, None, None] + 0.35 + 0.15
    collision = ((1.0 - hh).clamp(min=0).square()
                 + (bb_limit - bb).clamp(min=0).square()
                 + 0.5 * ((hb01_limit - hb01).clamp(min=0).square()
                          + (hb10_limit - hb10).clamp(min=0).square())).mean(dim=-1)
    safe = ((hh >= 1).all(dim=-1) & (bb >= bb_limit).all(dim=-1)
            & (hb01 >= hb01_limit).all(dim=-1) & (hb10 >= hb10_limit).all(dim=-1))
    duration = torch.where(active[:, None], future["duration"],
                           torch.zeros_like(future["duration"])).amax(dim=-1)
    length = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1).sum(dim=(-1, -2))
    rough = (speed[..., 1:] - speed[..., :-1]).square().mean(dim=(-1, -2))
    cost = 100 * collision + duration + 0.05 * length + 0.25 * rough
    cost = torch.where(valid, cost + (~safe).float() * 1e6,
                       torch.full_like(cost, float("inf")))
    index = cost.argmin(dim=-1)
    row = torch.arange(state.batch_size, device=path.device)
    return path[row, index], speed[row, index], valid[row, index], safe[row, index], index


def resample_plan(path, speed):
    """Interpolate XY and speed on the same 0.1m path-distance axis."""
    if path.shape != (*speed.shape, 2) or path.shape[1:] != (2, PATH_POINTS, 2):
        raise ValueError("expected selected path [B,2,33,2] and speed [B,2,33]")
    lengths = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
    cumulative = torch.cat((torch.zeros_like(lengths[..., :1]), lengths.cumsum(-1)), -1)
    end = cumulative[..., -1]
    query = torch.minimum(torch.arange(PATH_VERTICES, device=path.device) * PATH_DS,
                          end[..., None]).to(path.dtype)
    flat = cumulative.reshape(-1, PATH_POINTS).contiguous()
    query = query.reshape(-1, PATH_VERTICES).contiguous()
    lower = (torch.searchsorted(flat, query, right=True) - 1).clamp(0, PATH_POINTS - 2)
    ds = flat.gather(1, lower + 1) - flat.gather(1, lower)
    fraction = ((query - flat.gather(1, lower)) / ds.clamp(min=1e-8)).clamp(0, 1)
    values = torch.cat((path, speed[..., None]), dim=-1).reshape(-1, PATH_POINTS, 3)
    row = torch.arange(values.shape[0], device=path.device)[:, None]
    dense = values[row, lower] + fraction[..., None] * (values[row, lower + 1] - values[row, lower])
    dense = dense.reshape(path.shape[0], 2, PATH_VERTICES, 3)
    return dense[..., :2], dense[..., 2].clamp(MIN_SPEED, MAX_SPEED), end
