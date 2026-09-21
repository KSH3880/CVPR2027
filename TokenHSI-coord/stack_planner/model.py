"""Scene-token planner that corrects the previously committed XY path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Union

import torch
from torch import nn

from coordinator.geometry import (
    POSITION_SCALE, TOKEN_DIM, shared_to_world, state_to_tokens,
    world_to_shared,
)

from .schema import (
    AGENTS, MAX_SPEED, MIN_SPEED, STACK_CANDIDATES, STACK_PATH_DELTA_DIM,
    STACK_PATH_INPUT_DIM, STACK_PATH_POINTS, STACK_SPEED_DIM, CoordinatorState,
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
    delta_scale: float = 0.5
    retreat_delta_scale: float = 2.0
    path_update_alpha: float = 0.25
    history_steps: int = 1
    retreat_only: bool = False
    plain_carry: bool = False

    def __post_init__(self) -> None:
        if self.token_dim != TOKEN_DIM:
            raise ValueError(
                f"token_dim must preserve the existing input contract ({TOKEN_DIM})"
            )
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.encoder_layers <= 0:
            raise ValueError("Transformer layer count must be positive")
        if self.feedforward <= 0 or self.candidates <= 0:
            raise ValueError("feedforward and candidates must be positive")
        if self.delta_scale <= 0:
            raise ValueError("delta_scale must be positive")
        if self.retreat_delta_scale < self.delta_scale:
            raise ValueError("retreat_delta_scale must be at least delta_scale")
        if not 0.0 < self.path_update_alpha <= 1.0:
            raise ValueError("path_update_alpha must be in (0, 1]")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        if self.retreat_only and self.plain_carry:
            raise ValueError("retreat_only and plain_carry are exclusive")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class StackSceneEncoder(nn.Module):
    """Pool typed state tokens and the reference path into one scene token."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Linear(config.token_dim, config.d_model) for _ in range(3)
        ])
        self.entity_embedding = nn.Embedding(3, config.d_model)
        self.agent_embedding = nn.Embedding(AGENTS, config.d_model)
        self.scene_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.scene_token, std=0.02)
        self.plan_tokenizer = nn.Linear(STACK_PATH_INPUT_DIM, config.d_model)
        self.progress_tokenizer = nn.Linear(AGENTS, config.d_model)
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
            layer, num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.register_buffer(
            "entity_ids", torch.tensor([0, 1, 2] * AGENTS), persistent=False,
        )
        self.register_buffer(
            "agent_ids", torch.tensor([0, 0, 0, 1, 1, 1]), persistent=False,
        )

    def forward(self, tokens, history_valid, reference_path_local,
                previous_path_valid, path_progress):
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

        plan = self.plan_tokenizer(
            reference_path_local.reshape(batch, -1) / POSITION_SCALE
        )
        plan = plan + self.plan_valid_embedding(previous_path_valid.long())
        plan = plan + self.progress_tokenizer(
            path_progress / float(STACK_PATH_POINTS - 1)
        )
        scene = self.scene_token.expand(batch, -1, -1)
        value = torch.cat((scene, plan[:, None], value), dim=1)
        if history_valid is not None:
            if history_valid.shape != (batch, steps):
                raise ValueError("history_valid must be [B,history]")
            padding = (~history_valid[..., None].expand(
                -1, -1, entities
            )).reshape(batch, steps * entities)
            padding = torch.cat((
                torch.zeros(batch, 2, dtype=torch.bool, device=tokens.device),
                padding,
            ), dim=1)
        else:
            padding = None
        return self.transformer(value, src_key_padding_mask=padding)[:, 0]


def _smooth_delta(delta: torch.Tensor, passes: int = 2) -> torch.Tensor:
    """Low-pass a pointwise correction while keeping root correction zero."""
    for _ in range(passes):
        left = torch.cat((
            torch.zeros_like(delta[..., :1, :]), delta[..., :-1, :],
        ), dim=-2)
        right = torch.cat((
            delta[..., 1:, :], delta[..., -1:, :],
        ), dim=-2)
        delta = 0.25 * left + 0.50 * delta + 0.25 * right
        delta[..., 0, :] = 0.0
    return delta


def _smooth_speed(speed: torch.Tensor, passes: int = 2) -> torch.Tensor:
    """Low-pass a bounded pointwise speed profile without changing length."""
    for _ in range(passes):
        left = torch.cat((speed[..., :1], speed[..., :-1]), dim=-1)
        right = torch.cat((speed[..., 1:], speed[..., -1:]), dim=-1)
        speed = 0.25 * left + 0.50 * speed + 0.25 * right
    return speed


def _future_point_weight(path_progress: torch.Tensor,
                         previous_path_valid: torch.Tensor) -> torch.Tensor:
    """Continuous future mask with a two-point correction ramp at the root."""
    if path_progress.ndim != 2 or path_progress.shape[-1] != AGENTS:
        raise ValueError("path_progress must be [B,2]")
    index = torch.arange(
        STACK_PATH_POINTS, device=path_progress.device, dtype=path_progress.dtype,
    )
    future = ((index - path_progress[..., None]) / 2.0).clamp(0.0, 1.0)
    initial = (index > 0).to(path_progress.dtype).expand_as(future)
    return torch.where(previous_path_valid[:, None, None], future, initial)


class StackPlannerHeads(nn.Module):
    """Independent path-correction/speed proposals and evaluator."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__()
        self.config = config
        hidden = 2 * config.d_model

        def head(size):
            return nn.Sequential(
                nn.Linear(config.d_model, hidden), nn.GELU(),
                nn.Linear(hidden, size),
            )

        self.paths = nn.ModuleList([
            head(STACK_PATH_DELTA_DIM) for _ in range(config.candidates)
        ])
        self.speeds = nn.ModuleList([
            head(STACK_SPEED_DIM) for _ in range(config.candidates)
        ])
        self.candidate_evaluator = nn.Sequential(
            nn.Linear(
                config.d_model + STACK_PATH_INPUT_DIM + STACK_SPEED_DIM, hidden,
            ),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        self.value = head(1)
        for path in self.paths:
            nn.init.normal_(path[-1].weight, std=1e-3)
            nn.init.zeros_(path[-1].bias)
        for speed in self.speeds:
            nn.init.normal_(speed[-1].weight, std=1e-3)
            # Start close to the old MAX_SPEED execution behavior, while
            # retaining enough sigmoid gradient to learn slowdowns.
            nn.init.constant_(speed[-1].bias, 2.0)

    def decode_delta(self, base_path_local, previous_path_local,
                     path_delta_raw, speed_raw, path_point_weight):
        batch, candidates = path_delta_raw.shape[:2]
        expected = (batch, candidates, STACK_PATH_DELTA_DIM)
        if path_delta_raw.shape != expected:
            raise ValueError(f"path_delta_raw must be {expected}")
        speed_expected = (batch, candidates, STACK_SPEED_DIM)
        if speed_raw.shape != speed_expected:
            raise ValueError(f"speed_raw must be {speed_expected}")
        absolute_offset = torch.tanh(path_delta_raw).reshape(
            batch, candidates, AGENTS, STACK_PATH_POINTS - 1, 2,
        )
        absolute_offset = torch.cat((
            torch.zeros_like(absolute_offset[..., :1, :]), absolute_offset,
        ), dim=-2)
        absolute_offset = _smooth_delta(absolute_offset)
        offset_scale = absolute_offset.new_full(
            (1, 1, AGENTS, STACK_PATH_POINTS, 1), self.config.delta_scale,
        )
        if self.config.retreat_only:
            # The isolated diagnostic path is entirely A1 retreat. A2 is
            # masked below and remains a stationary part of the scene.
            offset_scale[..., 0, 1:, :] = self.config.retreat_delta_scale
        elif not self.config.plain_carry:
            # Only A1's post-goal suffix needs a wide workspace for retreat.
            offset_scale[..., 0, 22:, :] = self.config.retreat_delta_scale
        absolute_offset = (
            absolute_offset * offset_scale
            * path_point_weight[:, None, :, :, None]
        )
        if self.config.plain_carry:
            # Plain Carry has no post-placement retreat suffix.  Preserve the
            # physical pickup and placement anchors exactly while all other
            # points retain the same learned correction head.
            absolute_offset[..., 16, :] = 0.0
            absolute_offset[..., 32, :] = 0.0
        target_local = base_path_local[:, None] + absolute_offset
        blend = (
            self.config.path_update_alpha
            * path_point_weight[:, None, :, :, None]
        )
        path_local = previous_path_local[:, None] + blend * (
            target_local - previous_path_local[:, None]
        )
        path_local[..., 0, :] = previous_path_local[:, None, :, 0, :]
        update = path_local - previous_path_local[:, None]
        speed = speed_raw.reshape(
            batch, candidates, AGENTS, STACK_PATH_POINTS,
        )
        speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed)
        speed = _smooth_speed(speed)
        return path_local, update, speed

    def forward(self, scene, base_path_local, previous_path_local,
                path_point_weight):
        batch = scene.shape[0]
        path_delta_raw = torch.stack([path(scene) for path in self.paths], dim=1)
        speed_raw = torch.stack([speed(scene) for speed in self.speeds], dim=1)
        path_local, path_delta_local, speed = self.decode_delta(
            base_path_local, previous_path_local, path_delta_raw, speed_raw,
            path_point_weight,
        )
        expanded_scene = scene[:, None].expand(-1, path_local.shape[1], -1)
        evaluator_input = torch.cat((
            expanded_scene,
            (path_local.detach() / POSITION_SCALE).flatten(start_dim=2),
            ((speed.detach() - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).flatten(
                start_dim=2
            ),
        ), dim=-1)
        path_action_mask = path_point_weight[..., 1:, None].expand(
            -1, -1, -1, 2
        ).reshape(batch, -1) > 0
        if self.config.plain_carry:
            anchor = torch.ones(
                AGENTS, STACK_PATH_POINTS - 1, 2,
                dtype=torch.bool, device=scene.device,
            )
            anchor[:, 15, :] = False  # full-path pickup index 16
            anchor[:, 31, :] = False  # full-path goal index 32
            path_action_mask &= anchor.reshape(1, -1)
        return {
            "path_delta_raw": path_delta_raw,
            "speed_raw": speed_raw,
            "speed": speed,
            "path_delta_local": path_delta_local,
            "base_path_local": base_path_local,
            "reference_path_local": previous_path_local,
            "path_point_weight": path_point_weight,
            "path_action_mask": path_action_mask,
            "speed_action_mask": (
                path_point_weight.reshape(batch, STACK_SPEED_DIM) > 0
                if self.config.retreat_only else
                torch.ones(
                    batch, STACK_SPEED_DIM, dtype=torch.bool,
                    device=scene.device,
                )
            ),
            "candidate_logits": self.candidate_evaluator(
                evaluator_input
            ).squeeze(-1),
            "value": self.value(scene).squeeze(-1),
        }


class StackTrajectoryPlanner(nn.Module):
    """Predict bounded corrections to the last committed 33-point path."""

    def __init__(self, config: Optional[StackPlannerConfig] = None):
        super().__init__()
        self.config = config or StackPlannerConfig()
        self.scene_encoder = StackSceneEncoder(self.config)
        self.heads = StackPlannerHeads(self.config)

    @staticmethod
    def _geometric_reference_local(frame):
        root, box, goal = frame["root"], frame["box"], frame["goal"]
        t_first = torch.linspace(
            0.0, 1.0, 11, device=root.device, dtype=root.dtype,
        ).reshape(1, 1, 11, 1)
        t_second = torch.linspace(
            0.0, 1.0, 12, device=root.device, dtype=root.dtype,
        ).reshape(1, 1, 12, 1)
        first = root[..., None, :] + t_first * (box - root)[..., None, :]
        second = box[..., None, :] + t_second * (goal - box)[..., None, :]
        suffix = goal[..., None, :].expand(-1, -1, 12, -1)
        return torch.cat((
            first, second[..., 1:, :], suffix[..., 1:, :],
        ), dim=-2)

    @staticmethod
    def _plain_carry_reference_local(frame):
        root, box, goal = frame["root"], frame["box"], frame["goal"]
        t = torch.linspace(
            0.0, 1.0, 17, device=root.device, dtype=root.dtype,
        ).reshape(1, 1, 17, 1)
        approach = root[..., None, :] + t * (box - root)[..., None, :]
        carry = box[..., None, :] + t * (goal - box)[..., None, :]
        return torch.cat((approach, carry[..., 1:, :]), dim=-2)

    def _reference_path(self, state, previous_path_world, previous_path_valid,
                        base_path_world, base_path_valid, path_progress):
        _, frame = state_to_tokens(state)
        if self.config.retreat_only:
            geometric_local = frame["root"][..., None, :].expand(
                -1, -1, STACK_PATH_POINTS, -1,
            ).clone()
        elif self.config.plain_carry:
            geometric_local = self._plain_carry_reference_local(frame)
        else:
            geometric_local = self._geometric_reference_local(frame)
        geometric_world = shared_to_world(
            geometric_local, frame["center"], frame["angle"],
        )
        if previous_path_world is None:
            previous_path_world = torch.zeros_like(geometric_world)
            previous_path_valid = torch.zeros(
                state.batch_size, dtype=torch.bool, device=state.device,
            )
            base_path_world = torch.zeros_like(geometric_world)
            base_path_valid = torch.zeros_like(previous_path_valid)
        base_world = torch.where(
            base_path_valid[:, None, None, None],
            base_path_world, geometric_world,
        )
        reference_world = torch.where(
            previous_path_valid[:, None, None, None],
            previous_path_world, base_world,
        )
        base_local = world_to_shared(
            base_world, frame["center"], frame["angle"],
        )
        reference_local = world_to_shared(
            reference_world, frame["center"], frame["angle"],
        )
        point_weight = _future_point_weight(path_progress, previous_path_valid)
        if self.config.retreat_only:
            point_weight = point_weight.clone()
            point_weight[:, 1] = 0.0
        return frame, base_local, reference_local, point_weight

    def raw_heads(self, observation):
        history_valid = None
        previous_path_world = None
        base_path_world = None
        if hasattr(observation, "history_tokens"):
            observation.validate(self.config.history_steps)
            state = observation.state
            tokens = observation.history_tokens
            history_valid = observation.history_valid
            previous_path_world = observation.previous_path_world
            previous_path_valid = observation.previous_path_valid
            base_path_world = observation.base_path_world
            base_path_valid = observation.base_path_valid
            path_progress = observation.path_progress
        else:
            state = observation
            if not isinstance(state, CoordinatorState):
                state = CoordinatorState.from_mapping(state)
            tokens, _ = state_to_tokens(state)
            if self.config.history_steps > 1:
                raise ValueError(
                    "history-enabled planner requires StackPlannerObservation"
                )
            previous_path_valid = torch.zeros(
                state.batch_size, dtype=torch.bool, device=state.device,
            )
            base_path_valid = torch.zeros_like(previous_path_valid)
            path_progress = torch.zeros(
                state.batch_size, AGENTS, device=state.device,
                dtype=state.root_xy.dtype,
            )
        _, base_local, reference_local, point_weight = self._reference_path(
            state, previous_path_world, previous_path_valid,
            base_path_world, base_path_valid, path_progress,
        )
        scene = self.scene_encoder(
            tokens, history_valid, reference_local, previous_path_valid,
            path_progress,
        )
        return state, self.heads(
            scene, base_local, reference_local, point_weight,
        )

    def decode_delta(self, state, base_path_local, reference_path_local,
                     path_delta_raw, speed_raw, path_point_weight):
        _, frame = state_to_tokens(state)
        path_local, path_delta_local, speed = self.heads.decode_delta(
            base_path_local, reference_path_local, path_delta_raw, speed_raw,
            path_point_weight,
        )
        path_world = shared_to_world(
            path_local, frame["center"][:, None], frame["angle"][:, None],
        )
        base_path_world = shared_to_world(
            base_path_local, frame["center"], frame["angle"],
        )
        return {
            "path_local": path_local,
            "path_world": path_world,
            "speed": speed,
            "path_delta_local": path_delta_local,
            "base_path_world": base_path_world,
            "base_path_local": base_path_local,
            "reference_path_local": reference_path_local,
            "path_point_weight": path_point_weight,
        }

    def forward(self, state: Union[CoordinatorState, Mapping[str, torch.Tensor]]):
        current, raw = self.raw_heads(state)
        selected = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(current.batch_size, device=current.device)
        output = self.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            raw["path_delta_raw"][batch, selected][:, None],
            raw["speed_raw"][batch, selected][:, None],
            raw["path_point_weight"],
        )
        # Correction/reference tensors are training internals. Keep the public
        # Keep correction/reference tensors internal; path and speed are the
        # public execution contract.
        output.pop("path_delta_local")
        output.pop("base_path_local")
        output.pop("reference_path_local")
        output.pop("path_point_weight")
        output["selected_candidate"] = selected
        output["candidate_logits"] = raw["candidate_logits"]
        return output


__all__ = [
    "StackPlannerConfig", "StackTrajectoryPlanner", "_future_point_weight",
]
