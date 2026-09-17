"""Scene-token Transformer with independent full-path proposal heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Union

import torch
from torch import nn

from coordinator.geometry import TOKEN_DIM, shared_to_world, state_to_tokens

from .schema import (
    AGENTS,
    STACK_CANDIDATES,
    STACK_PATH_PARAM_DIM,
    CoordinatorState,
)


@dataclass(frozen=True)
class StackPlannerConfig:
    token_dim: int = TOKEN_DIM
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    feedforward: int = 256
    dropout: float = 0.0
    candidates: int = STACK_CANDIDATES
    residual_scale: float = 3.0
    delta_scale: float = 0.5
    history_steps: int = 1

    def __post_init__(self) -> None:
        if self.token_dim != TOKEN_DIM:
            raise ValueError(f"token_dim must preserve the existing input contract ({TOKEN_DIM})")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.encoder_layers <= 0:
            raise ValueError("Transformer layer count must be positive")
        if self.feedforward <= 0 or self.candidates <= 0:
            raise ValueError("feedforward and candidates must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.delta_scale <= 0:
            raise ValueError("delta_scale must be positive")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class StackSceneEncoder(nn.Module):
    """Pool typed observation tokens into one learned scene token."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        self.config = config
        # Root, box and goal carry different physical semantics.  Tokenize
        # each type independently while sharing the tokenizer across agents
        # and history slots.
        self.tokenizers = nn.ModuleList([
            nn.Linear(config.token_dim, config.d_model) for _ in range(3)
        ])
        self.entity_embedding = nn.Embedding(3, config.d_model)
        self.agent_embedding = nn.Embedding(AGENTS, config.d_model)
        self.scene_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.scene_token, std=0.02)
        self.plan_tokenizer = nn.Linear(STACK_PATH_PARAM_DIM, config.d_model)
        self.plan_valid_embedding = nn.Embedding(2, config.d_model)
        self.history_steps = config.history_steps
        if self.history_steps > 1:
            self.time_embedding = nn.Embedding(self.history_steps, config.d_model)
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

    def forward(self, tokens: torch.Tensor, history_valid=None,
                previous_path_raw=None, previous_path_valid=None) -> torch.Tensor:
        if tokens.ndim == 3:
            tokens = tokens[:, None]
        if tokens.ndim != 4 or tokens.shape[1] != self.history_steps:
            raise ValueError("scene tokens must be [B,history,6,token_dim]")
        batch, steps, entities, _ = tokens.shape
        value = torch.empty(
            batch, steps, entities, self.tokenizers[0].out_features,
            device=tokens.device, dtype=tokens.dtype,
        )
        for entity, tokenizer in enumerate(self.tokenizers):
            value[:, :, entity::3] = tokenizer(tokens[:, :, entity::3])
        value = value + self.entity_embedding(self.entity_ids)[None, None]
        value = value + self.agent_embedding(self.agent_ids)[None, None]
        if self.history_steps > 1:
            time_ids = torch.arange(steps, device=tokens.device)
            value = value + self.time_embedding(time_ids)[None, :, None]
        value = value.reshape(batch, steps * entities, -1)
        padding = None
        if history_valid is not None:
            if history_valid.shape != (batch, steps):
                raise ValueError("history_valid must be [B,history]")
            padding = (~history_valid[..., None].expand(-1, -1, entities)).reshape(
                batch, steps * entities
            )
        scene = self.scene_token.expand(batch, -1, -1)
        if previous_path_raw is None:
            previous_path_raw = tokens.new_zeros(batch, STACK_PATH_PARAM_DIM)
        if previous_path_valid is None:
            previous_path_valid = torch.zeros(
                batch, device=tokens.device, dtype=torch.bool,
            )
        if previous_path_raw.shape != (batch, STACK_PATH_PARAM_DIM):
            raise ValueError("previous_path_raw shape mismatch")
        if previous_path_valid.shape != (batch,):
            raise ValueError("previous_path_valid shape mismatch")
        plan = self.plan_tokenizer(previous_path_raw)
        plan = plan + self.plan_valid_embedding(previous_path_valid.long())
        value = torch.cat((scene, plan[:, None], value), dim=1)
        if padding is not None:
            padding = torch.cat((
                torch.zeros(batch, 2, dtype=torch.bool, device=tokens.device),
                padding,
            ), dim=1)
        encoded = self.transformer(value, src_key_padding_mask=padding)
        return encoded[:, 0]


class StackPlannerHeads(nn.Module):
    """Independent complete-path proposals plus a learned path evaluator."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        self.config = config
        hidden = 2 * config.d_model

        def head(size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, size),
            )

        self.path_dim = STACK_PATH_PARAM_DIM
        # Every head emits the same complete contract: both carry routes and
        # A1's retreat suffix.  Heads do not own individual path segments.
        self.paths = nn.ModuleList([
            head(self.path_dim) for _ in range(config.candidates)
        ])
        # Score actual proposal parameters, rather than a candidate ID.  The
        # proposal is detached here so selector gradients cannot improve a
        # score by distorting an unexecuted path head.
        self.candidate_evaluator = nn.Sequential(
            nn.Linear(config.d_model + self.path_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.value = head(1)
        # Tiny independent proposal initialization keeps every path close to
        # the executable straight prior without making all heads identical.
        for path in self.paths:
            nn.init.normal_(path[-1].weight, std=1e-3)
            nn.init.zeros_(path[-1].bias)

    def forward(self, scene: torch.Tensor,
                previous_path_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        path_delta_raw = torch.stack([path(scene) for path in self.paths], dim=1)
        path_raw = (
            previous_path_raw[:, None]
            + self.config.delta_scale * torch.tanh(path_delta_raw)
        ).clamp(-5.0, 5.0)
        expanded_scene = scene[:, None].expand(-1, path_raw.shape[1], -1)
        evaluator_input = torch.cat((expanded_scene, path_raw.detach()), dim=-1)
        return {
            "path_delta_raw": path_delta_raw,
            "path_raw": path_raw,
            "base_path_raw": previous_path_raw,
            "candidate_logits": self.candidate_evaluator(
                evaluator_input
            ).squeeze(-1),
            "value": self.value(scene).squeeze(-1),
        }


class StackTrajectoryPlanner(nn.Module):
    """Predict one joint XY path; execution concerns stay outside the model."""

    def __init__(self, config: Optional[StackPlannerConfig] = None):
        super().__init__()
        self.config = config or StackPlannerConfig()
        self.scene_encoder = StackSceneEncoder(self.config)
        self.heads = StackPlannerHeads(self.config)

    def raw_heads(self, state):
        """Return all complete proposals, evaluator logits and state value."""
        history_valid = None
        previous_path_raw = None
        previous_path_valid = None
        if hasattr(state, "history_tokens"):
            observation = state
            observation.validate(self.config.history_steps)
            state = observation.state
            tokens = observation.history_tokens
            history_valid = observation.history_valid
            previous_path_raw = observation.previous_path_raw
            previous_path_valid = observation.previous_path_valid
        else:
            if not isinstance(state, CoordinatorState):
                state = CoordinatorState.from_mapping(state)
            tokens, _ = state_to_tokens(state)
            if self.config.history_steps > 1:
                raise ValueError("history-enabled planner requires StackPlannerObservation")
        if previous_path_raw is None:
            previous_path_raw = tokens.new_zeros(
                state.batch_size, STACK_PATH_PARAM_DIM,
            )
        elif previous_path_valid is not None:
            previous_path_raw = torch.where(
                previous_path_valid[:, None], previous_path_raw,
                torch.zeros_like(previous_path_raw),
            )
        scene = self.scene_encoder(
            tokens, history_valid, previous_path_raw, previous_path_valid,
        )
        return state, self.heads(scene, previous_path_raw)

    def combine_delta(self, base_path_raw, path_delta_raw):
        """Apply a bounded one-decision update to committed path parameters."""
        return (
            base_path_raw[:, None]
            + self.config.delta_scale * torch.tanh(path_delta_raw)
        ).clamp(-5.0, 5.0)

    def forward(
        self,
        state: Union[CoordinatorState, Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        state, raw = self.raw_heads(state)
        selected = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(state.batch_size, device=state.device)
        output = self.decode(state, {
            "path_raw": raw["path_raw"][batch, selected][:, None],
        })
        output["path_parameters"] = raw["path_raw"][batch, selected][:, None]
        output["selected_candidate"] = selected
        output["candidate_logits"] = raw["candidate_logits"]
        return output

    def decode(
        self,
        state: CoordinatorState,
        raw: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Decode network means or sampled PPO head values into trajectories."""
        _, frame = state_to_tokens(state)

        packed = raw["path_raw"]
        path_dim = STACK_PATH_PARAM_DIM
        if (packed.ndim != 3 or packed.shape[0] != state.batch_size
                or packed.shape[-1] != path_dim):
            raise ValueError(f"path_raw must be [B,K,{path_dim}]")
        candidates = packed.shape[1]
        path_raw = packed.reshape(
            state.batch_size, candidates, AGENTS * 6 + 3, 2
        )
        route_residual = self.config.residual_scale * torch.tanh(
            path_raw[..., :AGENTS * 6, :].reshape(
                state.batch_size, candidates, AGENTS, 6, 2
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
