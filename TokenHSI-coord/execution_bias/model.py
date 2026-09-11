"""Deterministic executor path-distortion model.

The complete joint plan is flattened deliberately: a local point predictor
cannot see that the frozen executor is cutting a corner toward a distant goal.
The public contract contains only planned ``(x, y, v)`` points and returns an
exact ``(dx, dy)`` residual at every corresponding nominal timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


BIAS_SCHEMA_VERSION = "tokenhsi-execution-bias-v1"


@dataclass(frozen=True)
class ExecutionBiasConfig:
    agents: int = 2
    points: int = 33
    hidden: int = 512
    residual_blocks: int = 3
    position_scale: float = 10.0
    speed_scale: float = 1.5

    def __post_init__(self) -> None:
        if self.agents <= 0 or self.points < 2:
            raise ValueError("agents must be positive and points must be at least two")
        if self.hidden <= 0 or self.residual_blocks <= 0:
            raise ValueError("hidden and residual_blocks must be positive")
        if self.position_scale <= 0.0 or self.speed_scale <= 0.0:
            raise ValueError("normalization scales must be positive")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


def _shared_plan_frame(plan_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an SE(2) frame shared by all agents in a joint plan.

    Agent zero's first non-degenerate direction defines the angle.  A shared
    frame, rather than one frame per agent, preserves inter-agent geometry.
    """
    origin = plan_xy[:, 0, 0]
    segments = plan_xy[:, 0, 1:] - plan_xy[:, 0, :-1]
    usable = segments.square().sum(dim=-1) > 1e-8
    first = usable.to(torch.int64).argmax(dim=-1)
    row = torch.arange(plan_xy.shape[0], device=plan_xy.device)
    direction = segments[row, first]
    has_direction = usable.any(dim=-1)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    direction = torch.where(has_direction[:, None], direction, fallback)
    angle = torch.atan2(direction[:, 1], direction[:, 0])
    return origin, angle


def _rotation(angle: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        (torch.stack((cosine, -sine), dim=-1),
         torch.stack((sine, cosine), dim=-1)),
        dim=-2,
    )


class ExecutionBiasMLP(nn.Module):
    """Map a complete joint 33-point plan to deterministic XY residuals."""

    def __init__(self, config: ExecutionBiasConfig | None = None):
        super().__init__()
        self.config = config or ExecutionBiasConfig()
        input_dim = self.config.agents * self.config.points * 3
        output_dim = self.config.agents * self.config.points * 2
        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.config.hidden),
            nn.SiLU(),
        ]
        layers.extend(
            ResidualBlock(self.config.hidden)
            for _ in range(self.config.residual_blocks)
        )
        layers.extend((nn.LayerNorm(self.config.hidden), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.error_head = nn.Linear(self.config.hidden, output_dim)

        # Before supervised fitting, the frozen correction model is exactly an
        # identity execution model: predicted_actual_xy == planned_xy.
        nn.init.zeros_(self.error_head.weight)
        nn.init.zeros_(self.error_head.bias)

    def _validate(self, plan_xyv: torch.Tensor) -> None:
        expected = (self.config.agents, self.config.points, 3)
        if plan_xyv.ndim != 4 or tuple(plan_xyv.shape[1:]) != expected:
            raise ValueError(
                f"plan_xyv must be [B,{expected[0]},{expected[1]},3], "
                f"got {tuple(plan_xyv.shape)}"
            )
        if not torch.is_floating_point(plan_xyv):
            raise TypeError("plan_xyv must be floating point")
        if not torch.isfinite(plan_xyv).all():
            raise ValueError("plan_xyv contains non-finite values")
        if (plan_xyv[..., 2] < 0.0).any():
            raise ValueError("planned speed must be non-negative")

    def canonicalize(
        self, plan_xyv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonicalize the plan and return the local-to-world rotation."""
        self._validate(plan_xyv)
        xy = plan_xyv[..., :2]
        origin, angle = _shared_plan_frame(xy)
        local_to_world = _rotation(angle)
        centered = xy - origin[:, None, None]
        # Coordinates are stored as row vectors.  The conventional matrix R
        # maps local column vectors to world columns, hence row_world @ R maps
        # world rows back to the local frame.
        local_xy = torch.einsum("baqi,bij->baqj", centered, local_to_world)
        normalized = torch.cat(
            (local_xy / self.config.position_scale,
             plan_xyv[..., 2:3] / self.config.speed_scale),
            dim=-1,
        )
        return normalized, local_to_world

    def forward(self, plan_xyv: torch.Tensor) -> torch.Tensor:
        """Return deterministic world-frame ``(dx, dy)`` with shape [B,A,P,2]."""
        normalized, local_to_world = self.canonicalize(plan_xyv)
        hidden = self.backbone(normalized.flatten(start_dim=1))
        local_error = self.error_head(hidden).reshape(
            plan_xyv.shape[0], self.config.agents, self.config.points, 2
        )
        # Point zero is the observed current position, not a future prediction.
        local_error = torch.cat((torch.zeros_like(local_error[..., :1, :]),
                                 local_error[..., 1:, :]), dim=-2)
        return torch.einsum(
            "baqi,bij->baqj", local_error, local_to_world.transpose(-1, -2)
        )

    def executed_path(self, plan_xyv: torch.Tensor) -> torch.Tensor:
        """Differentiable expected physical XY path used by planner costs."""
        return plan_xyv[..., :2] + self(plan_xyv)
