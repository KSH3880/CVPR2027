"""Four-action PPO view of the B0 simple joint coordinator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .schema import CoordinatorState
from .simple_model import SIMPLE_ACTION_DIM, SimpleJointCoordinator


class SimpleCoordinatorActorCritic(nn.Module):
    def __init__(
        self, coordinator: Optional[SimpleJointCoordinator] = None, init_std: float = 0.15
    ):
        super().__init__()
        self.coordinator = coordinator or SimpleJointCoordinator()
        self.action_log_std = nn.Parameter(
            torch.full((SIMPLE_ACTION_DIM,), float(torch.log(torch.tensor(init_std))))
        )

    def distribution(self, state: CoordinatorState) -> Tuple[Normal, torch.Tensor, dict]:
        output = self.coordinator(state)
        mean = output["simple_action"]
        std = self.action_log_std.exp().clamp(0.01, 1.0).expand_as(mean)
        value = output["candidate_value"].mean(dim=1)
        return Normal(mean, std), value, output

    def act(self, state: CoordinatorState, deterministic: bool = False):
        distribution, value, template = self.distribution(state)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        output = self.coordinator.decode(
            state,
            action,
            value,
        )
        return output, action, log_prob, value

    def evaluate(self, state: CoordinatorState, action: torch.Tensor):
        distribution, value, output = self.distribution(state)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value, output
