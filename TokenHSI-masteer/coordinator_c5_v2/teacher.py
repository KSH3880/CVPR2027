"""Non-stop CEM teacher for role- and maneuver-conditioned C5 plans."""

from dataclasses import dataclass

import torch

from coordinator.planner import candidate_costs
from coordinator.schema import MAX_SPEED

from .core import (
    ACTION_DIM,
    AGENTS,
    KNOTS,
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
    C5TrajectoryCodec,
    canonical_action,
)


@dataclass(frozen=True)
class CEMConfig:
    population: int = 256
    elites: int = 16
    iterations: int = 5
    initial_std: float = 0.8
    min_std: float = 0.05
    role_weight: float = 2.0
    mode_weight: float = 4.0
    detour_margin: float = 0.15

    def __post_init__(self):
        if self.population <= 0 or not 0 < self.elites <= self.population:
            raise ValueError("invalid CEM population/elites")
        if self.iterations <= 0 or self.initial_std <= 0 or self.min_std <= 0:
            raise ValueError("invalid CEM iteration/std")


@dataclass
class TeacherResult:
    action: torch.Tensor
    path_world: torch.Tensor
    speed: torch.Tensor
    score: torch.Tensor
    diagnostics: dict
    role: torch.Tensor
    maneuver: torch.Tensor
    found_safe: torch.Tensor
    found_valid: torch.Tensor


def _gather(value, index):
    expand = index
    while expand.ndim < value.ndim:
        expand = expand.unsqueeze(-1)
    expand = expand.expand(*index.shape, *value.shape[index.ndim:])
    return value.gather(1, expand)


class CEMTeacher:
    def __init__(self, codec=None, config=None):
        self.codec = codec or C5TrajectoryCodec()
        self.config = config or CEMConfig()

    def _condition_penalty(self, action, decoded, role, maneuver):
        control = action.reshape(*action.shape[:-1], KNOTS, AGENTS, 3)
        batch, population = action.shape[:2]
        row = torch.arange(batch, device=action.device)[:, None]
        candidate = torch.arange(population, device=action.device)[None]
        priority = role[:, None].expand(-1, population)
        yielder = 1 - priority
        priority_control = control[row, candidate, :, priority]
        priority_residual = torch.tanh(priority_control[..., :2]).square().mean(dim=(-1, -2))
        priority_speed = decoded["speed"][row, candidate, priority]
        priority_slow = (MAX_SPEED - priority_speed).square().mean(dim=-1)
        penalty = self.config.role_weight * (priority_residual + priority_slow)

        yielding_control = control[row, candidate, :, yielder]
        lateral = torch.tanh(yielding_control[..., 1]).mean(dim=-1)
        left = maneuver[:, None] == MANEUVER_LEFT
        right = maneuver[:, None] == MANEUVER_RIGHT
        slow = maneuver[:, None] == MANEUVER_SLOW
        signed = torch.where(left, lateral, -lateral)
        detour_penalty = torch.relu(self.config.detour_margin - signed).square()
        straight_penalty = torch.tanh(yielding_control[..., :2]).square().mean(dim=(-1, -2))
        mode_penalty = torch.where(left | right, detour_penalty, torch.zeros_like(penalty))
        mode_penalty = torch.where(slow, straight_penalty, mode_penalty)
        return penalty + self.config.mode_weight * mode_penalty

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
            decoded = self.codec(action, state)
            diagnostics = candidate_costs(
                decoded, state, measured_executor_timing=True
            )
            score = diagnostics["rank_cost"] + self._condition_penalty(
                action, decoded, role, maneuver
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


def oracle_summary(results):
    if not results:
        raise ValueError("results must be non-empty")
    batch = results[0].found_safe.shape[0]
    if any(result.found_safe.shape[0] != batch for result in results):
        raise ValueError("teacher result batch sizes differ")
    safe = torch.stack([result.found_safe for result in results]).any(dim=0)
    valid = torch.stack([result.found_valid for result in results]).any(dim=0)
    return {
        "scenarios": batch,
        "oracle_safe": int(safe.sum().item()),
        "oracle_safe_rate": float(safe.float().mean().item()),
        "unresolved_nonstop": int((~safe).sum().item()),
        "valid_but_unsafe": int((valid & ~safe).sum().item()),
        "no_valid_plan": int((~valid).sum().item()),
    }
