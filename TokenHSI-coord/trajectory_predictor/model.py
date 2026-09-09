"""Small joint entity Transformer and anchored trajectory decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

from .geometry import TOKEN_DIM, baseline_path_local, shared_to_world, state_to_tokens
from .schema import AGENTS, COARSE_POINTS, PlannerState


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = TOKEN_DIM
    d_model: int = 128
    nhead: int = 4
    layers: int = 2
    feedforward: int = 256
    dropout: float = 0.0
    residual_scale: float = 3.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class InputNormalizer(nn.Module):
    """Checkpointed feature statistics, one set per entity position."""

    def __init__(self, tokens: int = AGENTS * 3, features: int = TOKEN_DIM):
        super().__init__()
        self.register_buffer("mean", torch.zeros(tokens, features))
        self.register_buffer("std", torch.ones(tokens, features))

    @torch.no_grad()
    def fit(self, values: torch.Tensor) -> None:
        if values.ndim != 3 or values.shape[1:] != self.mean.shape:
            raise ValueError(f"normalizer expected [B,{self.mean.shape[0]},{self.mean.shape[1]}]")
        self.mean.copy_(values.mean(dim=0))
        self.std.copy_(values.std(dim=0, unbiased=False).clamp(min=1e-4))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def payload(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @torch.no_grad()
    def load_payload(self, payload: Dict[str, torch.Tensor]) -> None:
        if set(payload) != {"mean", "std"}:
            raise ValueError("normalizer payload must contain exactly mean/std")
        if payload["mean"].shape != self.mean.shape or payload["std"].shape != self.std.shape:
            raise ValueError("normalizer shape mismatch")
        self.mean.copy_(payload["mean"].to(self.mean))
        self.std.copy_(payload["std"].to(self.std))


class JointTrajectoryPredictor(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.normalizer = InputNormalizer()
        self.input_proj = nn.Linear(c.token_dim, c.d_model)
        self.entity_embedding = nn.Embedding(3, c.d_model)
        self.agent_embedding = nn.Embedding(AGENTS, c.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.feedforward,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.layers, norm=nn.LayerNorm(c.d_model))
        joint_dim = AGENTS * 3 * c.d_model
        hidden = c.d_model * 2
        # 30 internal residual points per agent. Anchors 0/16/32 are never decoded.
        self.path_decoder = nn.Sequential(
            nn.Linear(joint_dim, hidden), nn.GELU(), nn.Linear(hidden, AGENTS * 30 * 2)
        )
        self.speed_decoder = nn.Sequential(
            nn.Linear(joint_dim, hidden), nn.GELU(), nn.Linear(hidden, AGENTS * 4 * 4)
        )

        entity_ids = torch.tensor([0, 1, 2] * AGENTS, dtype=torch.long)
        agent_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        self.register_buffer("entity_ids", entity_ids, persistent=False)
        self.register_buffer("agent_ids", agent_ids, persistent=False)

    def fit_normalizer(self, tokens: torch.Tensor) -> None:
        self.normalizer.fit(tokens)

    def forward(self, state: Union[PlannerState, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not isinstance(state, PlannerState):
            state = PlannerState.from_mapping(state)
        tokens, frame = state_to_tokens(state)
        x = self.input_proj(self.normalizer(tokens))
        x = x + self.entity_embedding(self.entity_ids)[None] + self.agent_embedding(self.agent_ids)[None]
        encoded = self.encoder(x)
        joint = encoded.reshape(encoded.shape[0], -1)

        residual_raw = self.path_decoder(joint).reshape(-1, AGENTS, 30, 2)
        residual_raw = torch.tanh(residual_raw) * self.config.residual_scale
        residual = torch.zeros(
            state.batch_size, AGENTS, COARSE_POINTS, 2,
            device=joint.device, dtype=joint.dtype,
        )
        residual[:, :, 1:16] = residual_raw[:, :, :15]
        residual[:, :, 17:32] = residual_raw[:, :, 15:]
        # Once carrying, the root and box are approximately coincident. Keeping
        # the unused approach leg on the analytic baseline prevents arbitrary
        # unconstrained loops from entering the inherited dense path buffer.
        carry = state.phase >= 0.5
        residual[:, :, 1:16] = torch.where(
            carry[:, :, None, None], torch.zeros_like(residual[:, :, 1:16]), residual[:, :, 1:16]
        )
        coarse_local = baseline_path_local(frame) + residual
        coarse_world = shared_to_world(coarse_local, frame["center"], frame["angle"])
        speed_logits = self.speed_decoder(joint).reshape(-1, AGENTS, 4, 4)
        return {
            "coarse_local": coarse_local,
            "coarse_world": coarse_world,
            "speed_logits": speed_logits,
        }
