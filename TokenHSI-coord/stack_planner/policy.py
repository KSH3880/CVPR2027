"""PPO policy wrapper for the isolated stack Transformer planner."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from coordinator.geometry import state_to_tokens
from coordinator.schema import AGENTS, CoordinatorState

from .model import StackTrajectoryPlanner


def _sizes(model: StackTrajectoryPlanner) -> Tuple[int, int]:
    k = model.config.candidates
    path = k * (AGENTS * 6 + 3) * 2
    return path, path


def _raw_heads(model: StackTrajectoryPlanner, state: CoordinatorState) -> Dict[str, torch.Tensor]:
    tokens, _ = state_to_tokens(state)
    return model.heads(model.candidate_decoder(model.scene_encoder(tokens)))


def pack_mean(model: StackTrajectoryPlanner, raw: Dict[str, torch.Tensor]) -> torch.Tensor:
    return raw["path_raw"].flatten(start_dim=1)


def decode_action(
    model: StackTrajectoryPlanner,
    state: CoordinatorState,
    action: torch.Tensor,
    template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    path_n, total = _sizes(model)
    if action.shape != (state.batch_size, total):
        raise ValueError(f"expected action [B,{total}], got {tuple(action.shape)}")
    sizes = (path_n,)
    names = ("path_raw",)
    raw = dict(template)
    start = 0
    for name, size in zip(names, sizes):
        raw[name] = action[:, start:start + size]
        start += size
    # Bounding happens in the differentiable path decoder.  A finite
    # clamp only prevents extreme exploration samples from saturating it.
    for name in names:
        raw[name] = raw[name].clamp(-5.0, 5.0)
    return model.decode(state, raw)


class StackPlannerActorCritic(nn.Module):
    def __init__(self, planner: StackTrajectoryPlanner, curve_std: float = 0.12,
                 endpoint_std: float = 0.20, anchor_std: float = 0.03):
        super().__init__()
        if min(curve_std, endpoint_std, anchor_std) <= 0.0:
            raise ValueError("planner exploration stds must be positive")
        self.planner = planner
        path_n, self.action_dim = _sizes(planner)
        # Keep box/goal anchors precise, but give Bézier controls enough range
        # to discover actual curved avoidance paths.  The old uniform 0.03
        # path std moved a control by only about 9 cm after residual scaling,
        # so PPO almost exclusively explored straight endpoint changes.
        std = torch.full((self.action_dim,), float(anchor_std))
        path_per_candidate = (AGENTS * 6 + 3) * 2
        for candidate in range(planner.config.candidates):
            base = candidate * path_per_candidate
            for agent in range(AGENTS):
                controls = base + (agent * 6 + 2) * 2
                std[controls:controls + 8] = float(curve_std)
            suffix = base + AGENTS * 6 * 2
            std[suffix:suffix + 2] = float(endpoint_std)
            std[suffix + 2:suffix + 6] = float(curve_std)
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
