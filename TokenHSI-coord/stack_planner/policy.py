"""PPO policy wrapper for the isolated stack Transformer planner."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from coordinator.geometry import state_to_tokens
from coordinator.schema import ACCEL_KNOTS, AGENTS, CoordinatorState

from .model import StackTrajectoryPlanner


def _sizes(model: StackTrajectoryPlanner) -> Tuple[int, int, int, int, int, int]:
    k = model.config.candidates
    ordinary_path = k * AGENTS * 4 * 2
    ordinary_accel = k * AGENTS * ACCEL_KNOTS
    dwell = k * AGENTS
    retreat_path = k * AGENTS * 3 * 2
    retreat_accel = k * AGENTS * ACCEL_KNOTS
    return (ordinary_path, ordinary_accel, dwell, retreat_path,
            retreat_accel, ordinary_path + ordinary_accel + dwell
            + retreat_path + retreat_accel)


def _raw_heads(model: StackTrajectoryPlanner, state: CoordinatorState) -> Dict[str, torch.Tensor]:
    tokens, _ = state_to_tokens(state)
    return model.heads(model.candidate_decoder(model.scene_encoder(tokens)))


def pack_mean(model: StackTrajectoryPlanner, raw: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((
        raw["path_raw"].flatten(start_dim=1),
        raw["accel_raw"].flatten(start_dim=1),
        raw["dwell_raw"].flatten(start_dim=1),
        raw["retreat_path_raw"].flatten(start_dim=1),
        raw["retreat_accel_raw"].flatten(start_dim=1),
    ), dim=-1)


def decode_action(
    model: StackTrajectoryPlanner,
    state: CoordinatorState,
    action: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    path_n, accel_n, dwell_n, retreat_n, retreat_accel_n, total = _sizes(model)
    if action.shape != (state.batch_size, total):
        raise ValueError(f"expected action [B,{total}], got {tuple(action.shape)}")
    sizes = (path_n, accel_n, dwell_n, retreat_n, retreat_accel_n)
    names = ("path_raw", "accel_raw", "dwell_raw",
             "retreat_path_raw", "retreat_accel_raw")
    raw = dict(template)
    start = 0
    for name, size in zip(names, sizes):
        raw[name] = action[:, start:start + size]
        start += size
    # Bounding happens in the differentiable trajectory decoder.  A finite
    # clamp only prevents extreme exploration samples from saturating it.
    for name in names:
        raw[name] = raw[name].clamp(-5.0, 5.0)
    return model.decode(state, raw)


class StackPlannerActorCritic(nn.Module):
    def __init__(self, planner: StackTrajectoryPlanner, init_std: float = 0.15):
        super().__init__()
        self.planner = planner
        path_n, accel_n, dwell_n, retreat_n, retreat_accel_n, self.action_dim = _sizes(planner)
        std = torch.full((self.action_dim,), float(init_std))
        # Ordinary path perturbations should initially preserve a usable Carry
        # plan; retreat endpoint exploration must remain large enough to move.
        std[:path_n] = 0.03
        retreat_start = path_n + accel_n + dwell_n
        std[retreat_start:retreat_start + retreat_n] = 0.08
        endpoint_per_candidate = AGENTS * 3 * 2
        for candidate in range(planner.config.candidates):
            base = retreat_start + candidate * endpoint_per_candidate
            for agent in range(AGENTS):
                pos = base + agent * 6
                std[pos:pos + 2] = 0.35
        self.action_log_std = nn.Parameter(std.log())

    def distribution(self, state: CoordinatorState):
        raw = _raw_heads(self.planner, state)
        mean = pack_mean(self.planner, raw)
        std = self.action_log_std.exp().clamp(0.005, 1.0).expand_as(mean)
        value = raw["candidate_value"].mean(dim=1)
        return Normal(mean, std), value, raw

    def act(self, state: CoordinatorState, deterministic: bool = False):
        distribution, value, raw = self.distribution(state)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return decode_action(self.planner, state, action, raw), action, log_prob, value

    def evaluate(self, state: CoordinatorState, action: torch.Tensor):
        distribution, value, raw = self.distribution(state)
        return (distribution.log_prob(action).sum(dim=-1),
                distribution.entropy().sum(dim=-1), value,
                decode_action(self.planner, state, action, raw))


__all__ = ["StackPlannerActorCritic", "decode_action", "pack_mean"]
