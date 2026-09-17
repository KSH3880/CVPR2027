"""Explicit decision-history observations for the stack planner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from coordinator.geometry import TOKENS, TOKEN_DIM, state_to_tokens
from coordinator.schema import STATE_KEYS, CoordinatorState


@dataclass
class StackPlannerObservation:
    """Current physical state plus right-aligned past decision tokens."""

    state: CoordinatorState
    history_tokens: torch.Tensor  # [B,H,6,12], includes current state
    history_valid: torch.Tensor   # [B,H]

    def validate(self, history_steps=None):
        batch = self.state.batch_size
        expected_tail = (TOKENS, TOKEN_DIM)
        if (self.history_tokens.ndim != 4
                or self.history_tokens.shape[0] != batch
                or tuple(self.history_tokens.shape[2:]) != expected_tail):
            raise ValueError("history_tokens must be [B,H,6,12]")
        if self.history_valid.shape != self.history_tokens.shape[:2]:
            raise ValueError("history_valid must be [B,H]")
        if history_steps is not None and self.history_tokens.shape[1] != history_steps:
            raise ValueError("history length does not match planner config")
        if self.history_valid.dtype != torch.bool:
            raise ValueError("history_valid must be bool")
        if not self.history_valid[:, -1].all():
            raise ValueError("current history slot must always be valid")

    @property
    def batch_size(self):
        return self.state.batch_size

    def index(self, index):
        return StackPlannerObservation(
            self.state.index(index), self.history_tokens[index],
            self.history_valid[index],
        )

    def clone(self):
        return StackPlannerObservation(
            self.state.clone(), self.history_tokens.clone(),
            self.history_valid.clone(),
        )


class StackHistoryBuffer:
    """Per-environment FIFO updated only at planner decision boundaries."""

    def __init__(self, batch_size, history_steps, device, dtype=torch.float32):
        if history_steps < 1:
            raise ValueError("history_steps must be positive")
        self.tokens = torch.zeros(
            batch_size, history_steps, TOKENS, TOKEN_DIM,
            device=device, dtype=dtype,
        )
        self.valid = torch.zeros(
            batch_size, history_steps, device=device, dtype=torch.bool,
        )

    @property
    def history_steps(self):
        return self.tokens.shape[1]

    def reset(self, env_mask):
        env_mask = torch.as_tensor(env_mask, device=self.tokens.device, dtype=torch.bool)
        self.tokens[env_mask] = 0.0
        self.valid[env_mask] = False

    def observe(self, state, reset_mask=None, commit=True):
        tokens_source = self.tokens
        valid_source = self.valid
        if reset_mask is not None:
            reset_mask = torch.as_tensor(
                reset_mask, device=self.tokens.device, dtype=torch.bool,
            )
            if commit:
                self.reset(reset_mask)
            else:
                # Counterfactual branches must be able to bootstrap from a
                # terminal state without mutating the single committed
                # history shared by all candidates.
                tokens_source = self.tokens.clone()
                valid_source = self.valid.clone()
                tokens_source[reset_mask] = 0.0
                valid_source[reset_mask] = False
        current, _ = state_to_tokens(state)
        tokens = torch.cat((tokens_source[:, 1:], current[:, None]), dim=1)
        valid = torch.cat((
            valid_source[:, 1:], torch.ones_like(valid_source[:, :1]),
        ), dim=1)
        observation = StackPlannerObservation(state, tokens, valid)
        observation.validate(self.history_steps)
        if commit:
            self.tokens.copy_(tokens)
            self.valid.copy_(valid)
        return observation


def flatten_observations(observations):
    if not observations:
        raise ValueError("cannot flatten empty observations")
    return StackPlannerObservation(
        CoordinatorState(**{
            key: torch.cat([getattr(item.state, key) for item in observations], dim=0)
            for key in STATE_KEYS
        }),
        torch.cat([item.history_tokens for item in observations], dim=0),
        torch.cat([item.history_valid for item in observations], dim=0),
    )


__all__ = ["StackHistoryBuffer", "StackPlannerObservation", "flatten_observations"]
