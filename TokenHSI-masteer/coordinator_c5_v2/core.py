"""Standalone 48-D C5 V2 actor and trajectory codec."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from coordinator.geometry import rotate_xy, state_to_tokens
from coordinator.schema import AGENTS, MAX_SPEED, MIN_SPEED, PATH_POINTS, CoordinatorState


KNOTS = 8
ACTION_DIM = KNOTS * AGENTS * 3
STATE_DIM = AGENTS * 3 * 12
ROLE_COUNT = AGENTS

MANEUVER_STRAIGHT = 0
MANEUVER_LEFT = 1
MANEUVER_RIGHT = 2
MANEUVER_SLOW = 3
MANEUVER_COUNT = 4


def encode_state(state):
    tokens, _ = state_to_tokens(state)
    return tokens.flatten(start_dim=1)


def _labels(value, batch, classes, name):
    if value.shape != (batch,) or value.dtype == torch.bool or value.is_floating_point():
        raise ValueError(f"{name} must be an integer [B] tensor")
    if ((value < 0) | (value >= classes)).any():
        raise ValueError(f"{name} values must be in [0, {classes - 1}]")
    return F.one_hot(value.long(), classes).to(dtype=torch.float32)


@dataclass(frozen=True)
class C5Config:
    history_frames: int = 8
    hidden_dim: int = 128
    residual_scale: float = 1.0
    fixed_pickup_dwell: float = 1.5

    def __post_init__(self):
        if self.history_frames <= 0 or self.hidden_dim <= 0:
            raise ValueError("history_frames and hidden_dim must be positive")
        if self.residual_scale <= 0 or self.fixed_pickup_dwell < 0:
            raise ValueError("invalid codec scale or pickup dwell")


class C5HistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = STATE_DIM + ROLE_COUNT + MANEUVER_COUNT
        self.gru = nn.GRU(width, config.hidden_dim, batch_first=True)

    def forward(self, history, role, maneuver):
        if history.ndim != 3 or history.shape[-1] != STATE_DIM:
            raise ValueError(f"history must be [B,T,{STATE_DIM}]")
        batch, frames = history.shape[:2]
        role_onehot = _labels(role, batch, ROLE_COUNT, "role").to(history)
        maneuver_onehot = _labels(
            maneuver, batch, MANEUVER_COUNT, "maneuver"
        ).to(history)
        condition = torch.cat((role_onehot, maneuver_onehot), dim=-1)
        condition = condition[:, None].expand(-1, frames, -1)
        _, hidden = self.gru(torch.cat((history, condition), dim=-1))
        return hidden[-1]


class C5ProposalActor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or C5Config()
        self.encoder = C5HistoryEncoder(self.config)
        self.body = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(self.config.hidden_dim, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.6))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            self.mean.bias.reshape(KNOTS, AGENTS, 3)[..., 2] = -4.0

    def distribution(self, history, role, maneuver):
        context = self.encoder(history, role, maneuver)
        mean = self.mean(self.body(context))
        return mean, self.log_std.exp().expand_as(mean)

    def sample(self, history, role, maneuver, candidates, generator=None):
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        mean, std = self.distribution(history, role, maneuver)
        noise = torch.randn(
            mean.shape[0], candidates, ACTION_DIM,
            device=mean.device, dtype=mean.dtype, generator=generator,
        )
        return mean[:, None] + std[:, None] * noise


class C5TrajectoryCodec(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or C5Config()

    @staticmethod
    def _interpolate(value, length):
        shape = value.shape
        flat = value.movedim(-2, -1).reshape(-1, shape[-1], shape[-2])
        dense = F.interpolate(flat, size=length, mode="linear", align_corners=True)
        return dense.reshape(*shape[:-2], shape[-1], length).movedim(-1, -2)

    def forward(self, action, state):
        if action.ndim != 3 or action.shape[-1] != ACTION_DIM:
            raise ValueError(f"action must be [B,N,{ACTION_DIM}]")
        if action.shape[0] != state.batch_size:
            raise ValueError("action and state batch sizes differ")
        batch, candidates = action.shape[:2]
        control = action.reshape(batch, candidates, KNOTS, AGENTS, 3)

        root = state.root_xy[:, None]
        box = state.box_xyz[..., :2][:, None]
        goal = state.goal_xy[:, None]
        t = torch.linspace(0.0, 1.0, 17, device=action.device, dtype=action.dtype)
        t = t.reshape(1, 1, 1, 17, 1)
        approach = root[..., None, :] + t * (box - root)[..., None, :]
        carry = box[..., None, :] + t * (goal - box)[..., None, :]
        base = torch.cat((approach, carry[..., 1:, :]), dim=-2)
        base = base.expand(-1, candidates, -1, -1, -1).clone()

        residual = control[..., :2].permute(0, 1, 3, 2, 4)
        legs = []
        for leg in (residual[..., :4, :], residual[..., 4:, :]):
            zero = torch.zeros_like(leg[..., :1, :])
            dense = self._interpolate(torch.cat((zero, leg, zero), dim=-2), 17)
            legs.append(dense[..., 1:-1, :])
        held = state.held[:, None, :, None, None] >= 0.5
        legs[0] = torch.where(held, torch.zeros_like(legs[0]), legs[0])
        residual33 = torch.cat(
            (
                torch.zeros_like(base[..., :1, :]),
                legs[0],
                torch.zeros_like(base[..., :1, :]),
                legs[1],
                torch.zeros_like(base[..., :1, :]),
            ),
            dim=-2,
        )
        angle = state.heading[:, 0][:, None, None, None]
        residual_world = rotate_xy(torch.tanh(residual33), angle)
        path = base + self.config.residual_scale * residual_world
        path[..., 0, :] = root
        path[..., 16, :] = box
        path[..., 32, :] = goal

        speed_raw = control[..., 2].permute(0, 1, 3, 2).unsqueeze(-1)
        speed_raw = self._interpolate(speed_raw, PATH_POINTS).squeeze(-1)
        speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed_raw)
        dwell = torch.full(
            (batch, candidates, AGENTS), self.config.fixed_pickup_dwell,
            device=action.device, dtype=action.dtype,
        )
        dwell = torch.where(state.held[:, None] >= 0.5, torch.zeros_like(dwell), dwell)
        return {"path_world": path, "speed": speed, "pickup_dwell": dwell}


def _speed_raw(speed, device, dtype):
    ratio = (MAX_SPEED - float(speed)) / (MAX_SPEED - MIN_SPEED)
    ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
    return torch.tensor(
        torch.logit(torch.tensor(ratio)).item(), device=device, dtype=dtype
    )


def canonical_action(role, maneuver, residual_scale=1.0, detour=0.65, slow_speed=0.55):
    batch = role.shape[0]
    _labels(role, batch, ROLE_COUNT, "role")
    _labels(maneuver, batch, MANEUVER_COUNT, "maneuver")
    action = torch.zeros(batch, KNOTS, AGENTS, 3, device=role.device, dtype=torch.float32)
    action[..., 2] = _speed_raw(MAX_SPEED - 1e-3, role.device, action.dtype)
    yielder = 1 - role.long()
    row = torch.arange(batch, device=role.device)
    slow = maneuver == MANEUVER_SLOW
    action[row[slow], :, yielder[slow], 2] = _speed_raw(
        slow_speed, role.device, action.dtype
    )
    lateral = min(float(detour) / float(residual_scale), 0.95)
    lateral_raw = torch.atanh(torch.tensor(lateral, device=role.device))
    left = maneuver == MANEUVER_LEFT
    right = maneuver == MANEUVER_RIGHT
    action[row[left], :, yielder[left], 1] = lateral_raw
    action[row[right], :, yielder[right], 1] = -lateral_raw
    return action.reshape(batch, ACTION_DIM)
