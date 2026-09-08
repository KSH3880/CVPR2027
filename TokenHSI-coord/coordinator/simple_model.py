"""B0: one tiny joint MLP producing one safe-by-construction plan.

The model has no learned candidates or transformer.  It predicts only one
carry-path lateral offset and one slowdown control per agent.  The pickup path
is always the analytic straight line used by the strong baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import state_to_tokens
from .schema import AGENTS, CANDIDATES, MAX_SPEED, MIN_SPEED, CoordinatorState


SIMPLE_ACTION_DIM = AGENTS * 2


@dataclass(frozen=True)
class SimpleCoordinatorConfig:
    hidden: int = 128
    max_lateral: float = 1.0
    safe_bow: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _build_plan(
    state: CoordinatorState,
    action: torch.Tensor,
    value: torch.Tensor,
    max_lateral: float,
    safe_bow: bool,
) -> Dict[str, torch.Tensor]:
    if action.shape != (state.batch_size, SIMPLE_ACTION_DIM):
        raise ValueError(
            f"simple action must be [B,{SIMPLE_ACTION_DIM}], got {tuple(action.shape)}"
        )
    if safe_bow:
        carry_distance = (state.goal_xy - state.box_xyz[..., :2]).norm(dim=-1)
        lateral_limit = carry_distance.clamp(max=max_lateral)
        lateral = torch.tanh(action[:, :AGENTS]) * lateral_limit
    else:
        lateral = action[:, :AGENTS].clamp(-max_lateral, max_lateral)
    # Zero is exactly the analytic 1.5 m/s baseline.  Only negative actions
    # request slowdown; the executor applies its existing temporal rate limit.
    slowdown = torch.tanh(F.relu(-action[:, AGENTS:]))
    target_speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * slowdown

    root = state.root_xy
    box = state.box_xyz[..., :2]
    goal = state.goal_xy
    t = torch.linspace(0.0, 1.0, 17, device=root.device, dtype=root.dtype)
    first = root[:, :, None] + t[None, None, :, None] * (box - root)[:, :, None]
    delta = goal - box
    normal = torch.stack((-delta[..., 1], delta[..., 0]), dim=-1)
    normal = normal / delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    bow = 4.0 * t * (1.0 - t)
    second = (
        box[:, :, None]
        + t[None, None, :, None] * delta[:, :, None]
        + bow[None, None, :, None] * lateral[:, :, None, None] * normal[:, :, None]
    )
    path_single = torch.cat((first, second[:, :, 1:]), dim=2)

    first_speed = torch.full(
        (state.batch_size, AGENTS, 17), MAX_SPEED, device=root.device, dtype=root.dtype
    )
    second_speed = MAX_SPEED + t[None, None] * (target_speed - MAX_SPEED)[:, :, None]
    speed_single = torch.cat((first_speed, second_speed[:, :, 1:]), dim=2)

    path = path_single[:, None].expand(-1, CANDIDATES, -1, -1, -1)
    speed = speed_single[:, None].expand(-1, CANDIDATES, -1, -1)
    segment = path[..., 1:, :] - path[..., :-1, :]
    ds = segment.norm(dim=-1).clamp(min=1e-6)
    acceleration = (speed[..., 1:].square() - speed[..., :-1].square()) / (2.0 * ds)
    dwell_single = torch.where(
        state.held >= 0.5,
        torch.zeros_like(state.held),
        torch.full_like(state.held, 0.5),
    )

    residual_single = torch.zeros(
        state.batch_size, AGENTS, 4, 2, device=root.device, dtype=root.dtype
    )
    residual_single[:, :, 2:] = lateral[:, :, None, None] * normal[:, :, None]
    residual = residual_single[:, None].expand(-1, CANDIDATES, -1, -1, -1)
    accel_knots = torch.zeros(
        state.batch_size, CANDIDATES, AGENTS, 8, device=root.device, dtype=root.dtype
    )
    return {
        "path_local": path,
        "path_world": path,
        "speed": speed,
        "acceleration": acceleration,
        "pickup_dwell": dwell_single[:, None].expand(-1, CANDIDATES, -1),
        "candidate_value": value[:, None].expand(-1, CANDIDATES),
        "risk_logits": torch.zeros(
            state.batch_size, CANDIDATES, 3, device=root.device, dtype=root.dtype
        ),
        "control_residual": residual,
        "accel_knots_raw": accel_knots,
        "simple_action": action,
        "lateral": lateral,
        "target_speed": target_speed,
    }


class SimpleJointCoordinator(nn.Module):
    """Flattened state -> two lateral offsets + two slowdown controls."""

    def __init__(self, config: Optional[SimpleCoordinatorConfig] = None):
        super().__init__()
        self.config = config or SimpleCoordinatorConfig()
        token_dim = 6 * 12
        hidden = self.config.hidden
        self.backbone = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden, SIMPLE_ACTION_DIM)
        self.value_head = nn.Linear(hidden, 1)
        # Exact analytic plan at iteration zero.
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def encode(self, state: CoordinatorState) -> torch.Tensor:
        tokens, _ = state_to_tokens(state)
        return self.backbone(tokens.flatten(start_dim=1))

    def decode(
        self,
        state: CoordinatorState,
        action: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return _build_plan(
            state,
            action,
            value,
            self.config.max_lateral,
            self.config.safe_bow,
        )

    def forward(
        self, state: Union[CoordinatorState, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(state, CoordinatorState):
            state = CoordinatorState.from_mapping(state)
        hidden = self.encode(state)
        action = self.action_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return self.decode(state, action, value)
