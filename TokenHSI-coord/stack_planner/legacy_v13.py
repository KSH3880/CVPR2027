"""Read-only V13 planner semantics used by the interactive viewer.

V13 accumulated a bounded delta onto the last committed path.  Newer
checkpoints target a fixed episode path and therefore cannot faithfully load a
V13 state dict merely by supplying defaults for new config fields.
"""

from __future__ import annotations

from typing import Mapping, Union

import torch

from coordinator.geometry import (
    POSITION_SCALE, shared_to_world, state_to_tokens, world_to_shared,
)

from .model import (
    StackPlannerConfig, StackPlannerHeads, StackTrajectoryPlanner,
    _future_point_weight, _smooth_delta, _smooth_speed,
)
from .schema import (
    AGENTS, MAX_SPEED, MIN_SPEED, STACK_PATH_DELTA_DIM, STACK_PATH_POINTS,
    STACK_SPEED_DIM, CoordinatorState,
)


class StackPlannerHeadsV13(StackPlannerHeads):
    """V13's previous-path delta decoder with current parameter names."""

    def decode_delta(self, reference_path_local, path_delta_raw, speed_raw,
                     path_point_weight):
        batch, candidates = path_delta_raw.shape[:2]
        expected = (batch, candidates, STACK_PATH_DELTA_DIM)
        if path_delta_raw.shape != expected:
            raise ValueError(f"path_delta_raw must be {expected}")
        speed_expected = (batch, candidates, STACK_SPEED_DIM)
        if speed_raw.shape != speed_expected:
            raise ValueError(f"speed_raw must be {speed_expected}")
        delta = self.config.delta_scale * torch.tanh(path_delta_raw)
        delta = delta.reshape(
            batch, candidates, AGENTS, STACK_PATH_POINTS - 1, 2,
        )
        delta = torch.cat((torch.zeros_like(delta[..., :1, :]), delta), dim=-2)
        delta = _smooth_delta(delta)
        delta = delta * path_point_weight[:, None, :, :, None]
        path_local = reference_path_local[:, None] + delta
        path_local[..., 0, :] = reference_path_local[:, None, :, 0, :]
        speed = speed_raw.reshape(
            batch, candidates, AGENTS, STACK_PATH_POINTS,
        )
        speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed)
        speed = _smooth_speed(speed)
        return path_local, delta, speed

    def forward(self, scene, reference_path_local, path_point_weight):
        batch = scene.shape[0]
        path_delta_raw = torch.stack([path(scene) for path in self.paths], dim=1)
        speed_raw = torch.stack([speed(scene) for speed in self.speeds], dim=1)
        path_local, path_delta_local, speed = self.decode_delta(
            reference_path_local, path_delta_raw, speed_raw, path_point_weight,
        )
        expanded_scene = scene[:, None].expand(-1, path_local.shape[1], -1)
        evaluator_input = torch.cat((
            expanded_scene,
            (path_local.detach() / POSITION_SCALE).flatten(start_dim=2),
            ((speed.detach() - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).flatten(
                start_dim=2
            ),
        ), dim=-1)
        return {
            "path_delta_raw": path_delta_raw,
            "speed_raw": speed_raw,
            "speed": speed,
            "path_delta_local": path_delta_local,
            "reference_path_local": reference_path_local,
            "path_point_weight": path_point_weight,
            "path_action_mask": path_point_weight[..., 1:, None].expand(
                -1, -1, -1, 2
            ).reshape(batch, -1) > 0,
            "speed_action_mask": torch.ones(
                batch, STACK_SPEED_DIM, dtype=torch.bool, device=scene.device,
            ),
            "candidate_logits": self.candidate_evaluator(
                evaluator_input
            ).squeeze(-1),
            "value": self.value(scene).squeeze(-1),
        }


class StackTrajectoryPlannerV13(StackTrajectoryPlanner):
    """Exact V13 inference path; never used for training or resume."""

    def __init__(self, config: StackPlannerConfig):
        super().__init__(config)
        self.heads = StackPlannerHeadsV13(config)

    def _reference_path(self, state, previous_path_world, previous_path_valid,
                        path_progress):
        _, frame = state_to_tokens(state)
        geometric_local = self._geometric_reference_local(frame)
        geometric_world = shared_to_world(
            geometric_local, frame["center"], frame["angle"],
        )
        if previous_path_world is None:
            previous_path_world = torch.zeros_like(geometric_world)
            previous_path_valid = torch.zeros(
                state.batch_size, dtype=torch.bool, device=state.device,
            )
        reference_world = torch.where(
            previous_path_valid[:, None, None, None],
            previous_path_world, geometric_world,
        )
        reference_local = world_to_shared(
            reference_world, frame["center"], frame["angle"],
        )
        point_weight = _future_point_weight(path_progress, previous_path_valid)
        return frame, reference_local, point_weight

    def raw_heads(self, observation):
        history_valid = None
        previous_path_world = None
        if hasattr(observation, "history_tokens"):
            observation.validate(self.config.history_steps)
            state = observation.state
            tokens = observation.history_tokens
            history_valid = observation.history_valid
            previous_path_world = observation.previous_path_world
            previous_path_valid = observation.previous_path_valid
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
            path_progress = torch.zeros(
                state.batch_size, AGENTS, device=state.device,
                dtype=state.root_xy.dtype,
            )
        _, reference_local, point_weight = self._reference_path(
            state, previous_path_world, previous_path_valid, path_progress,
        )
        scene = self.scene_encoder(
            tokens, history_valid, reference_local, previous_path_valid,
            path_progress,
        )
        return state, self.heads(scene, reference_local, point_weight)

    def decode_delta(self, state, reference_path_local, path_delta_raw,
                     speed_raw, path_point_weight):
        _, frame = state_to_tokens(state)
        path_local, path_delta_local, speed = self.heads.decode_delta(
            reference_path_local, path_delta_raw, speed_raw, path_point_weight,
        )
        path_world = shared_to_world(
            path_local, frame["center"][:, None], frame["angle"][:, None],
        )
        return {
            "path_local": path_local,
            "path_world": path_world,
            "speed": speed,
            "path_delta_local": path_delta_local,
            "reference_path_local": reference_path_local,
            "path_point_weight": path_point_weight,
        }

    def forward(self, state: Union[CoordinatorState, Mapping[str, torch.Tensor]]):
        current, raw = self.raw_heads(state)
        selected = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(current.batch_size, device=current.device)
        output = self.decode_delta(
            current, raw["reference_path_local"],
            raw["path_delta_raw"][batch, selected][:, None],
            raw["speed_raw"][batch, selected][:, None],
            raw["path_point_weight"],
        )
        output.pop("path_delta_local")
        output.pop("reference_path_local")
        output.pop("path_point_weight")
        output["selected_candidate"] = selected
        output["candidate_logits"] = raw["candidate_logits"]
        return output


__all__ = ["StackTrajectoryPlannerV13"]
