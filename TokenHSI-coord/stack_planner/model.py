"""Transformer planner emitting one end-to-end XY path per stack agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Union

import torch
from torch import nn

from coordinator.geometry import TOKEN_DIM, shared_to_world, state_to_tokens

from .schema import (
    AGENTS,
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
    """One path head for the complete joint future."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        hidden = 2 * config.d_model

        def head(size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, size),
            )

        # Six vectors per agent describe root->box->goal. Three more extend
        # Agent 1 to the final point of the same path; this is not a separate
        # retreat output/head.
        self.path = head((AGENTS * 6 + 3) * 2)
        self.value = head(1)
        # A fresh planner starts from straight paths. PPO exploration, rather
        # than arbitrary last-layer initialization, owns the first deviations
        # from that executable prior.
        nn.init.zeros_(self.path[-1].weight)
        nn.init.zeros_(self.path[-1].bias)

    def forward(self, decoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "path_raw": self.path(decoded),
            "candidate_value": self.value(decoded).squeeze(-1),
        }


class StackTrajectoryPlanner(nn.Module):
    """Predict one joint XY path; execution concerns stay outside the model."""

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

        path_raw = raw["path_raw"].reshape(
            state.batch_size, self.config.candidates, AGENTS * 6 + 3, 2
        )
        route_residual = self.config.residual_scale * torch.tanh(
            path_raw[..., :AGENTS * 6, :].reshape(
                state.batch_size, self.config.candidates, AGENTS, 6, 2
            )
        )
        suffix_raw = path_raw[..., AGENTS * 6:, :]
        root = frame["root"][:, None]
        junction = frame["box"][:, None] + route_residual[..., 0, :]
        route_endpoint = frame["goal"][:, None] + route_residual[..., 1, :]
        first_delta = junction - root
        second_delta = route_endpoint - junction
        base_control = torch.stack((
            root + first_delta / 3.0,
            root + 2.0 * first_delta / 3.0,
            junction + second_delta / 3.0,
            junction + 2.0 * second_delta / 3.0,
        ), dim=-2)
        control = base_control + route_residual[..., 2:, :]

        def cubic(start, controls, end, samples):
            curve_t = torch.linspace(
                0.0, 1.0, samples, device=start.device, dtype=start.dtype,
            ).reshape((1,) * (start.ndim - 1) + (samples, 1))
            omt = 1.0 - curve_t
            return (
                omt ** 3 * start.unsqueeze(-2)
                + 3.0 * omt ** 2 * curve_t * controls[..., 0, :].unsqueeze(-2)
                + 3.0 * omt * curve_t ** 2 * controls[..., 1, :].unsqueeze(-2)
                + curve_t ** 3 * end.unsqueeze(-2)
            )

        first_curve = cubic(root, control[..., :2, :], junction, 11)
        second_curve = cubic(junction, control[..., 2:, :], route_endpoint, 12)
        a1_start = route_endpoint[..., 0, :]
        a1_delta = self.config.residual_scale * torch.tanh(suffix_raw[..., 0, :])
        a1_endpoint = a1_start + a1_delta
        a1_base = torch.stack((
            a1_start + a1_delta / 3.0,
            a1_start + 2.0 * a1_delta / 3.0,
        ), dim=-2)
        a1_control = a1_base + self.config.residual_scale * torch.tanh(
            suffix_raw[..., 1:, :]
        )
        a1_suffix = cubic(a1_start, a1_control, a1_endpoint, 12)
        a2_suffix = route_endpoint[..., 1, None, :].expand(-1, -1, 12, -1)
        suffix = torch.stack((a1_suffix, a2_suffix), dim=2)
        path_local = torch.cat((
            first_curve, second_curve[..., 1:, :], suffix[..., 1:, :],
        ), dim=-2)
        path_local[..., 0, :] = root
        path_world = shared_to_world(
            path_local,
            frame["center"][:, None],
            frame["angle"][:, None],
        )

        return {
            "path_local": path_local,
            "path_world": path_world,
        }
