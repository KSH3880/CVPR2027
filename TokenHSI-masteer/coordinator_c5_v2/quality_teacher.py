"""CEM teacher with mode discipline and executable-path efficiency costs."""

from dataclasses import dataclass

import torch

from coordinator.planner import candidate_costs
from coordinator.schema import AGENTS

from .core import (
    ACTION_DIM,
    KNOTS,
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
    canonical_action,
)
from .teacher import CEMConfig, CEMTeacher, TeacherResult, _gather


@dataclass(frozen=True)
class QualityCEMConfig(CEMConfig):
    role_weight: float = 20.0
    mode_weight: float = 20.0
    detour_weight: float = 30.0
    backtrack_weight: float = 100.0
    smoothness_weight: float = 10.0
    max_detour_ratio: float = 1.35


class QualityCEMTeacher(CEMTeacher):
    def __init__(self, codec=None, config=None):
        super().__init__(codec=codec, config=config or QualityCEMConfig())

    @staticmethod
    def _project_mode(action, role, maneuver):
        control = action.reshape(*action.shape[:-1], KNOTS, AGENTS, 3).clone()
        batch = action.shape[0]
        agent = torch.arange(AGENTS, device=action.device).reshape(1, 1, 1, AGENTS)
        yielder = (1 - role).reshape(batch, 1, 1, 1)
        is_yielder = agent == yielder
        left = (maneuver == MANEUVER_LEFT).reshape(batch, 1, 1, 1)
        right = (maneuver == MANEUVER_RIGHT).reshape(batch, 1, 1, 1)
        lateral = control[..., 1]
        lateral = torch.where(left & is_yielder, lateral.abs(), lateral)
        lateral = torch.where(right & is_yielder, -lateral.abs(), lateral)
        control[..., 1] = lateral
        slow = (maneuver == MANEUVER_SLOW).reshape(batch, 1, 1, 1, 1)
        control[..., :2] = torch.where(
            slow, torch.zeros_like(control[..., :2]), control[..., :2]
        )
        return control.reshape_as(action)

    def _quality_penalty(self, action, decoded, state, role, maneuver):
        penalty = super()._condition_penalty(action, decoded, role, maneuver)
        path = decoded["path_world"]
        segment = path[..., 1:, :] - path[..., :-1, :]
        path_length = segment.norm(dim=-1).sum(dim=-1)
        direct = (
            (state.box_xyz[..., :2] - state.root_xy).norm(dim=-1)
            + (state.goal_xy - state.box_xyz[..., :2]).norm(dim=-1)
        ).clamp_min(1e-4)
        detour_ratio = path_length / direct[:, None]
        detour = torch.relu(
            detour_ratio - self.config.max_detour_ratio
        ).square().sum(dim=-1)

        approach = state.box_xyz[..., :2] - state.root_xy
        carry = state.goal_xy - state.box_xyz[..., :2]
        approach = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        carry = carry / carry.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        directions = torch.cat((
            approach[:, None, :, None].expand(-1, path.shape[1], -1, 16, -1),
            carry[:, None, :, None].expand(-1, path.shape[1], -1, 16, -1),
        ), dim=-2)
        progress = (segment * directions).sum(dim=-1)
        backtrack = torch.relu(-progress).square().mean(dim=(-1, -2))
        second = path[..., 2:, :] - 2.0 * path[..., 1:-1, :] + path[..., :-2, :]
        second[..., 14:17, :] = 0.0
        smoothness = second.square().sum(dim=-1).mean(dim=(-1, -2))
        return (
            penalty
            + self.config.detour_weight * detour
            + self.config.backtrack_weight * backtrack
            + self.config.smoothness_weight * smoothness
        )

    @torch.no_grad()
    def search(self, state, role, maneuver, generator=None):
        batch = state.batch_size
        if role.shape != (batch,) or maneuver.shape != (batch,):
            raise ValueError("role and maneuver must be [B]")
        mean = canonical_action(
            role, maneuver, residual_scale=self.codec.config.residual_scale
        ).to(device=state.device, dtype=state.root_xy.dtype)
        std = torch.full_like(mean, self.config.initial_std)
        found_safe = torch.zeros(batch, dtype=torch.bool, device=state.device)
        found_valid = torch.zeros_like(found_safe)
        last = None
        for _ in range(self.config.iterations):
            noise = torch.randn(
                batch, self.config.population, ACTION_DIM,
                device=state.device, dtype=state.root_xy.dtype, generator=generator,
            )
            noise[:, 0] = 0.0
            action = mean[:, None] + std[:, None] * noise
            action = self._project_mode(action, role, maneuver)
            decoded = self.codec(action, state)
            diagnostics = candidate_costs(
                decoded, state, measured_executor_timing=True
            )
            score = diagnostics["rank_cost"] + self._quality_penalty(
                action, decoded, state, role, maneuver
            )
            found_safe |= (diagnostics["safe"] & diagnostics["valid"]).any(dim=1)
            found_valid |= diagnostics["valid"].any(dim=1)
            elite_index = score.argsort(dim=1)[:, :self.config.elites]
            elite_action = _gather(action, elite_index)
            mean = elite_action.mean(dim=1)
            std = elite_action.std(dim=1, unbiased=False).clamp_min(self.config.min_std)
            last = action, decoded, diagnostics, score, elite_index

        action, decoded, diagnostics, score, elite_index = last
        elite_diagnostics = {
            key: _gather(value, elite_index)
            for key, value in diagnostics.items()
            if value.ndim >= 2 and value.shape[:2] == score.shape
        }
        return TeacherResult(
            action=_gather(action, elite_index),
            path_world=_gather(decoded["path_world"], elite_index),
            speed=_gather(decoded["speed"], elite_index),
            score=_gather(score, elite_index),
            diagnostics=elite_diagnostics,
            role=role,
            maneuver=maneuver,
            found_safe=found_safe,
            found_valid=found_valid,
        )
