"""Transformer-based joint path and speed model for the stack task.

V1 intentionally preserves the existing coordinator I/O.  Stack-specific
event, placement-height, and dependency heads can be added to
``StackPlannerHeads`` later without changing the scene encoder or trajectory
decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Union

import torch
from torch import nn

from coordinator.geometry import (
    TOKEN_DIM,
    build_bezier_paths,
    integrate_speed,
    shared_to_world,
    state_to_tokens,
)

from .schema import (
    ACCEL_KNOTS,
    AGENTS,
    MAX_SPEED,
    PATH_POINTS,
    STACK_CANDIDATES,
    CoordinatorState,
)


@dataclass(frozen=True)
class StackPlannerConfig:
    token_dim: int = TOKEN_DIM
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.0
    candidates: int = STACK_CANDIDATES
    residual_scale: float = 3.0
    retreat_distance: float = 2.0

    def __post_init__(self) -> None:
        if self.token_dim != TOKEN_DIM:
            raise ValueError(f"token_dim must preserve the existing input contract ({TOKEN_DIM})")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("Transformer layer counts must be positive")
        if self.feedforward <= 0 or self.candidates <= 0:
            raise ValueError("feedforward and candidates must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.retreat_distance <= 0:
            raise ValueError("retreat_distance must be positive")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class StackSceneEncoder(nn.Module):
    """Encode the unchanged six root/box/goal entity tokens."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.token_dim, config.d_model)
        self.entity_embedding = nn.Embedding(3, config.d_model)
        self.agent_embedding = nn.Embedding(AGENTS, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.register_buffer(
            "entity_ids", torch.tensor([0, 1, 2] * AGENTS), persistent=False
        )
        self.register_buffer(
            "agent_ids", torch.tensor([0, 0, 0, 1, 1, 1]), persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(tokens)
        value = value + self.entity_embedding(self.entity_ids)[None]
        value = value + self.agent_embedding(self.agent_ids)[None]
        return self.transformer(value)


class StackCandidateDecoder(nn.Module):
    """Decode unnamed coordinated futures from learned candidate queries."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.queries = nn.Parameter(torch.empty(config.candidates, config.d_model))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        query = self.queries[None].expand(memory.shape[0], -1, -1)
        return self.transformer(query, memory)


class StackPlannerHeads(nn.Module):
    """Current path/speed contract and the extension point for later heads."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        hidden = 2 * config.d_model

        def head(size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, size),
            )

        self.path = head(AGENTS * 4 * 2)
        self.acceleration = head(AGENTS * ACCEL_KNOTS)
        self.pickup_dwell = head(AGENTS)
        # endpoint delta + two cubic control-point residuals.  This is a
        # physical root trajectory, not a virtual Carry-object prediction.
        self.retreat_path = head(AGENTS * 3 * 2)
        self.retreat_acceleration = head(AGENTS * ACCEL_KNOTS)
        self.value = head(1)
        self.risk = head(3)
        # A fresh planner starts from straight paths and constant speed.  PPO
        # exploration, rather than arbitrary last-layer initialization, owns
        # the first deviations from that executable prior.
        for module in (
            self.path, self.acceleration,
            self.retreat_path, self.retreat_acceleration,
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, decoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "path_raw": self.path(decoded),
            "accel_raw": self.acceleration(decoded),
            "dwell_raw": self.pickup_dwell(decoded),
            "retreat_path_raw": self.retreat_path(decoded),
            "retreat_accel_raw": self.retreat_acceleration(decoded),
            "candidate_value": self.value(decoded).squeeze(-1),
            "risk_logits": self.risk(decoded),
        }


class StackTrajectoryPlanner(nn.Module):
    """Predict joint XY paths and path-local speeds for two stack agents."""

    def __init__(self, config: Optional[StackPlannerConfig] = None):
        super().__init__()
        self.config = config or StackPlannerConfig()
        self.scene_encoder = StackSceneEncoder(self.config)
        self.candidate_decoder = StackCandidateDecoder(self.config)
        self.heads = StackPlannerHeads(self.config)

    def forward(
        self,
        state: Union[CoordinatorState, Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(state, CoordinatorState):
            state = CoordinatorState.from_mapping(state)

        tokens, frame = state_to_tokens(state)
        memory = self.scene_encoder(tokens)
        decoded = self.candidate_decoder(memory)
        raw = self.heads(decoded)

        return self.decode(state, raw)

    def decode(
        self,
        state: CoordinatorState,
        raw: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Decode network means or sampled PPO head values into trajectories."""
        _, frame = state_to_tokens(state)

        residual = raw["path_raw"].reshape(
            state.batch_size, self.config.candidates, AGENTS, 4, 2
        )
        residual = self.config.residual_scale * torch.tanh(residual)
        # A carried box makes root->box a completed leg.  Keeping that leg on
        # the baseline matches the existing coordinator execution contract.
        first_leg = torch.where(
            state.held[:, None, :, None, None] >= 0.5,
            torch.zeros_like(residual[..., :2, :]),
            residual[..., :2, :],
        )
        residual = torch.cat((first_leg, residual[..., 2:, :]), dim=-2)
        path_local = build_bezier_paths(frame, residual)
        path_world = shared_to_world(
            path_local,
            frame["center"][:, None],
            frame["angle"][:, None],
        )

        accel_raw = raw["accel_raw"].reshape(
            state.batch_size, self.config.candidates, AGENTS, ACCEL_KNOTS
        )
        initial_speed = state.root_vel_xy.norm(dim=-1)[:, None].expand(
            -1, self.config.candidates, -1
        )
        speed, acceleration = integrate_speed(path_world, accel_raw, initial_speed)

        pickup_dwell = 0.5 + 2.5 * torch.sigmoid(
            raw["dwell_raw"].reshape(
                state.batch_size, self.config.candidates, AGENTS
            )
        )
        pickup_dwell = torch.where(
            state.held[:, None] >= 0.5,
            torch.zeros_like(pickup_dwell),
            pickup_dwell,
        )

        retreat_raw = raw["retreat_path_raw"].reshape(
            state.batch_size, self.config.candidates, AGENTS, 3, 2
        )
        endpoint_delta = self.config.retreat_distance * torch.tanh(
            retreat_raw[..., 0, :]
        )
        root = frame["root"][:, None]
        endpoint = root + endpoint_delta
        control_residual = self.config.residual_scale * torch.tanh(
            retreat_raw[..., 1:, :]
        )
        base = torch.stack(
            (root + endpoint_delta / 3.0, root + 2.0 * endpoint_delta / 3.0),
            dim=-2,
        )
        control = base + control_residual
        # Use two 17-sample halves of the same cubic so the public path length
        # stays identical to the ordinary coordinator output (33 points).
        t = torch.linspace(
            0.0, 1.0, PATH_POINTS,
            device=state.device, dtype=state.root_xy.dtype,
        ).reshape(1, 1, 1, PATH_POINTS, 1)
        omt = 1.0 - t
        retreat_local = (
            omt ** 3 * root.unsqueeze(-2)
            + 3.0 * omt ** 2 * t * control[..., 0, :].unsqueeze(-2)
            + 3.0 * omt * t ** 2 * control[..., 1, :].unsqueeze(-2)
            + t ** 3 * endpoint.unsqueeze(-2)
        )
        retreat_local[..., 0, :] = root
        retreat_local[..., -1, :] = endpoint
        retreat_world = shared_to_world(
            retreat_local, frame["center"][:, None], frame["angle"][:, None]
        )
        retreat_accel_raw = raw["retreat_accel_raw"].reshape(
            state.batch_size, self.config.candidates, AGENTS, ACCEL_KNOTS
        )
        retreat_speed, retreat_acceleration = integrate_speed(
            retreat_world, retreat_accel_raw, initial_speed
        )

        return {
            "path_local": path_local,
            "path_world": path_world,
            "speed": speed,
            "acceleration": acceleration,
            "pickup_dwell": pickup_dwell,
            "candidate_value": raw["candidate_value"],
            "risk_logits": raw["risk_logits"],
            "control_residual": residual,
            "accel_knots_raw": accel_raw,
            "retreat_path_local": retreat_local,
            "retreat_path_world": retreat_world,
            "retreat_speed": retreat_speed,
            "retreat_acceleration": retreat_acceleration,
            "retreat_control_raw": retreat_raw,
            "retreat_accel_knots_raw": retreat_accel_raw,
            "trajectory": torch.cat((path_world, speed.unsqueeze(-1)), dim=-1),
        }
