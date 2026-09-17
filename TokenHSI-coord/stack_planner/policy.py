"""Continuous full-path heads evaluated by same-state counterfactual rollout."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.distributions import Normal

from coordinator.schema import AGENTS, CoordinatorState

from .model import StackTrajectoryPlanner
from .schema import STACK_PATH_DELTA_DIM, STACK_PATH_POINTS


def _path_dim() -> int:
    return STACK_PATH_DELTA_DIM


class StackPlannerActorCritic(nn.Module):
    """Sample full paths per head; evaluator selection is supervised."""

    def __init__(self, planner: StackTrajectoryPlanner, point_std: float = 0.12,
                 endpoint_std: float = 0.20, anchor_std: float = 0.03):
        super().__init__()
        if min(point_std, endpoint_std, anchor_std) <= 0.0:
            raise ValueError("planner exploration stds must be positive")
        self.planner = planner
        self.continuous_action_dim = _path_dim()
        # The leading index identifies which independent head produced the
        # path. It is supervised by full rollout comparison, not sampled as a
        # PPO categorical action.
        self.action_dim = self.continuous_action_dim + 1
        std = torch.full((AGENTS, STACK_PATH_POINTS - 1, 2), float(point_std))
        # Keep the points nearest the geometric box/goal anchors precise while
        # allowing the final retreat end to explore more broadly.
        std[:, 9, :] = float(anchor_std)   # full-path index 10
        std[:, 20, :] = float(anchor_std)  # full-path index 21
        std[:, -1, :] = float(endpoint_std)
        std = std.reshape(-1)
        # Exploration can specialize per independent full-path head.
        self.action_log_std = nn.Parameter(
            std.log()[None].repeat(planner.config.candidates, 1)
        )

    def distribution(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        std = self.action_log_std.exp().clamp(0.005, 1.0)
        std = std[None].expand(current.batch_size, -1, -1)
        paths = Normal(raw["path_delta_raw"], std)
        return paths, raw["value"], raw

    def value(self, state: CoordinatorState):
        _, raw = self.planner.raw_heads(state)
        return raw["value"]

    def mean_output(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        candidate = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(candidate.shape[0], device=candidate.device)
        output = self.planner.decode_delta(
            current, raw["reference_path_local"],
            raw["path_delta_raw"][batch, candidate][:, None],
        )
        output["selected_candidate"] = candidate
        output["candidate_logits"] = raw["candidate_logits"]
        return output

    def all_mean_outputs(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        output = self.planner.decode_delta(
            current, raw["reference_path_local"], raw["path_delta_raw"],
        )
        output["candidate_logits"] = raw["candidate_logits"]
        return output

    def act(self, state: CoordinatorState, deterministic: bool = False):
        paths, value, raw = self.distribution(state)
        # Runtime selection always belongs to the learned evaluator. During
        # training sample_all() evaluates every head instead.
        candidate = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(candidate.shape[0], device=candidate.device)
        mean = paths.loc[batch, candidate]
        std = paths.scale[batch, candidate]
        selected_distribution = Normal(mean, std)
        delta_action = (
            selected_distribution.mean if deterministic
            else selected_distribution.sample()
        )
        log_prob = selected_distribution.log_prob(delta_action).sum(dim=-1)
        packed_action = torch.cat(
            (candidate[:, None].to(delta_action), delta_action), dim=-1,
        )
        current = state.state if hasattr(state, "history_tokens") else state
        output = self.planner.decode_delta(
            current, raw["reference_path_local"], delta_action[:, None],
        )
        output["selected_candidate"] = candidate
        output["candidate_logits"] = raw["candidate_logits"]
        return output, packed_action, log_prob, value

    def sample_all(self, state: CoordinatorState):
        """Sample and decode every full-path head from the same scene."""
        paths, value, raw = self.distribution(state)
        delta_action = paths.sample()
        log_prob = paths.log_prob(delta_action).sum(dim=-1)
        candidate = torch.arange(
            self.planner.config.candidates, device=delta_action.device,
            dtype=delta_action.dtype,
        )[None, :, None].expand(delta_action.shape[0], -1, -1)
        packed_action = torch.cat((candidate, delta_action), dim=-1)
        current = state.state if hasattr(state, "history_tokens") else state
        output = self.planner.decode_delta(
            current, raw["reference_path_local"], delta_action,
        )
        output["candidate_logits"] = raw["candidate_logits"]
        return output, packed_action, log_prob, value

    def evaluate(self, state: CoordinatorState, action: torch.Tensor):
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"expected packed action [B,{self.action_dim}]")
        paths, value, raw = self.distribution(state)
        candidate_float = action[:, 0]
        candidate = candidate_float.long()
        if not torch.equal(candidate_float, candidate_float.round()):
            raise ValueError("candidate action must be an integer")
        if ((candidate < 0) | (candidate >= self.planner.config.candidates)).any():
            raise ValueError("candidate action out of range")
        delta_action = action[:, 1:]
        batch = torch.arange(candidate.shape[0], device=candidate.device)
        selected_distribution = Normal(
            paths.loc[batch, candidate], paths.scale[batch, candidate]
        )
        # Candidate selection is supervised from full counterfactual returns;
        # it is not part of the PPO action probability.
        log_prob = selected_distribution.log_prob(delta_action).sum(dim=-1)
        entropy = selected_distribution.entropy().sum(dim=-1)
        current = state.state if hasattr(state, "history_tokens") else state
        decoded = self.planner.decode_delta(
            current, raw["reference_path_local"], delta_action[:, None],
        )
        decoded["selected_candidate"] = candidate
        decoded["candidate_logits"] = raw["candidate_logits"]
        return log_prob, entropy, value, decoded

    def diversity(self, state: CoordinatorState, margin: float = 0.25):
        """Keep alternative full-path heads distinct in world-space metres."""
        output = self.all_mean_outputs(state)
        path = output["path_world"][:, :, 0]
        candidates = path.shape[1]
        if candidates < 2:
            zero = path.new_zeros(())
            return {"loss": zero, "distance": zero}
        pairs = torch.triu_indices(candidates, candidates, 1, device=path.device)
        difference = path[:, pairs[0]] - path[:, pairs[1]]
        distance = difference.square().sum(dim=-1).mean(dim=-1).clamp(min=1e-8).sqrt()
        return {
            "loss": (margin - distance).clamp(min=0.0).square().mean(),
            "distance": distance.mean(),
        }


__all__ = ["StackPlannerActorCritic"]
