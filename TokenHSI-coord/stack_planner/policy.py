"""Continuous full-path heads evaluated by same-state counterfactual rollout."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.distributions import Normal

from coordinator.schema import AGENTS, CoordinatorState

from .model import StackTrajectoryPlanner
from .schema import STACK_PATH_POINTS


class StackPlannerActorCritic(nn.Module):
    """Sample full paths per head; evaluator selection is supervised."""

    def __init__(self, planner: StackTrajectoryPlanner, point_std: float = 0.12,
                 endpoint_std: float = 0.20, anchor_std: float = 0.03,
                 speed_std: float = 0.20):
        super().__init__()
        if min(point_std, endpoint_std, anchor_std, speed_std) <= 0.0:
            raise ValueError("planner exploration stds must be positive")
        self.planner = planner
        self.path_action_dim = planner.heads.path_action_dim
        self.speed_action_dim = planner.heads.speed_action_dim
        self.continuous_action_dim = (
            self.path_action_dim + self.speed_action_dim
        )
        # The leading index identifies which independent head produced the
        # path. It is supervised by full rollout comparison, not sampled as a
        # PPO categorical action.
        self.action_dim = self.continuous_action_dim + 1
        if planner.config.plain_carry:
            std = torch.full(
                (self.path_action_dim,), float(point_std),
            )
        else:
            std = torch.full((AGENTS, STACK_PATH_POINTS - 1, 2), float(point_std))
            # Keep the points nearest the geometric box/goal anchors precise while
            # allowing the final retreat end to explore more broadly.
            std[:, 9, :] = float(anchor_std)   # full-path index 10
            std[:, 20, :] = float(anchor_std)  # full-path index 21
            std[:, -1, :] = float(endpoint_std)
            std = std.reshape(-1)
        std = torch.cat((
            std, torch.full((self.speed_action_dim,), float(speed_std)),
        ))
        # Exploration can specialize per independent full-path head.
        self.action_log_std = nn.Parameter(
            std.log()[None].repeat(planner.config.candidates, 1)
        )

    def distribution(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        std = self.action_log_std.exp().clamp(0.005, 1.0)
        std = std[None].expand(current.batch_size, -1, -1)
        mean = torch.cat((raw["path_delta_raw"], raw["speed_raw"]), dim=-1)
        actions = Normal(mean, std)
        return actions, raw["value"], raw

    def _split_action(self, action):
        return (
            action[..., :self.path_action_dim],
            action[..., self.path_action_dim:],
        )

    @staticmethod
    def _action_mask(raw, dtype):
        return torch.cat((
            raw["path_action_mask"], raw["speed_action_mask"],
        ), dim=-1).to(dtype)

    def value(self, state: CoordinatorState):
        _, raw = self.planner.raw_heads(state)
        return raw["value"]

    def mean_output(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        candidate = raw["candidate_logits"].argmax(dim=-1)
        batch = torch.arange(candidate.shape[0], device=candidate.device)
        output = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            raw["path_delta_raw"][batch, candidate][:, None],
            raw["speed_raw"][batch, candidate][:, None],
            raw["path_point_weight"],
        )
        output["selected_candidate"] = candidate
        output["candidate_logits"] = raw["candidate_logits"]
        return output

    def all_mean_outputs(self, state: CoordinatorState):
        current, raw = self.planner.raw_heads(state)
        output = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            raw["path_delta_raw"],
            raw["speed_raw"],
            raw["path_point_weight"],
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
        action_mask = self._action_mask(raw, delta_action.dtype)
        log_prob = (
            selected_distribution.log_prob(delta_action) * action_mask
        ).sum(dim=-1)
        packed_action = torch.cat(
            (candidate[:, None].to(delta_action), delta_action), dim=-1,
        )
        current = state.state if hasattr(state, "history_tokens") else state
        path_action, speed_action = self._split_action(delta_action)
        output = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            path_action[:, None],
            speed_action[:, None],
            raw["path_point_weight"],
        )
        output["selected_candidate"] = candidate
        output["candidate_logits"] = raw["candidate_logits"]
        return output, packed_action, log_prob, value

    def sample_all(self, state: CoordinatorState):
        """Decode sampled execution paths and noise-free reference paths."""
        paths, value, raw = self.distribution(state)
        delta_action = paths.sample()
        action_mask = self._action_mask(raw, delta_action.dtype)[:, None]
        log_prob = (paths.log_prob(delta_action) * action_mask).sum(dim=-1)
        candidate = torch.arange(
            self.planner.config.candidates, device=delta_action.device,
            dtype=delta_action.dtype,
        )[None, :, None].expand(delta_action.shape[0], -1, -1)
        packed_action = torch.cat((candidate, delta_action), dim=-1)
        current = state.state if hasattr(state, "history_tokens") else state
        path_action, speed_action = self._split_action(delta_action)
        output = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            path_action, speed_action,
            raw["path_point_weight"],
        )
        mean_path_action, mean_speed_action = self._split_action(paths.loc)
        mean_output = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            mean_path_action, mean_speed_action,
            raw["path_point_weight"],
        )
        # The sampled output is physically executed for on-policy credit. The
        # noise-free output is the recurrent planning reference, preventing
        # exploration noise from accumulating as a trajectory random walk.
        output["mean_path_local"] = mean_output["path_local"]
        output["mean_path_world"] = mean_output["path_world"]
        output["mean_speed"] = mean_output["speed"]
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
        action_mask = self._action_mask(raw, delta_action.dtype)
        log_prob = (
            selected_distribution.log_prob(delta_action) * action_mask
        ).sum(dim=-1)
        entropy = (selected_distribution.entropy() * action_mask).sum(dim=-1)
        current = state.state if hasattr(state, "history_tokens") else state
        path_action, speed_action = self._split_action(delta_action)
        decoded = self.planner.decode_delta(
            current, raw["base_path_local"], raw["reference_path_local"],
            path_action[:, None],
            speed_action[:, None],
            raw["path_point_weight"],
        )
        decoded["selected_candidate"] = candidate
        decoded["candidate_logits"] = raw["candidate_logits"]
        return log_prob, entropy, value, decoded

    def diversity(self, state: CoordinatorState, margin: float = 0.25,
                  output=None):
        """Keep coarse routes distinct without rewarding pointwise zigzags."""
        if output is None:
            output = self.all_mean_outputs(state)
        path = output["path_world"][:, :, 0]
        # Diversity on raw points can be satisfied by alternating left/right
        # corrections. Compare only a strongly low-passed, coarse route.
        coarse = path
        for _ in range(4):
            coarse = 0.25 * torch.cat((
                coarse[..., :1, :], coarse[..., :-1, :],
            ), dim=-2) + 0.50 * coarse + 0.25 * torch.cat((
                coarse[..., 1:, :], coarse[..., -1:, :],
            ), dim=-2)
        coarse = coarse[..., ::4, :]
        if self.planner.config.plain_carry:
            # Penalize curvature *changes*, not curvature itself. A broad,
            # consistently curved avoidance route should be free; alternating
            # turns/noisy S-curves should not. Split at pickup so the intended
            # approach-to-carry corner is not charged.
            full_path = output["path_world"]
            variations = []
            for leg in (full_path[..., :17, :], full_path[..., 16:, :]):
                segment = leg[..., 1:, :] - leg[..., :-1, :]
                direction = segment / segment.norm(dim=-1, keepdim=True).clamp(
                    min=1e-6,
                )
                turn = direction[..., 1:, :] - direction[..., :-1, :]
                variation = turn[..., 1:, :] - turn[..., :-1, :]
                variations.append(variation.square().sum(dim=-1).mean())
            smoothness = torch.stack(variations).mean()
        else:
            correction = output["path_delta_local"]
            second = (
                correction[..., 2:, :]
                - 2.0 * correction[..., 1:-1, :]
                + correction[..., :-2, :]
            )
            future_second = output["path_point_weight"][..., 2:]
            smoothness_numerator = (
                second.square().sum(dim=-1)
                * future_second[:, None]
            ).sum()
            smoothness_denominator = (
                future_second.sum()
                * correction.new_tensor(correction.shape[1])
            ).clamp(min=1.0)
            smoothness = smoothness_numerator / smoothness_denominator
        speed_difference = output["speed"][..., 1:] - output["speed"][..., :-1]
        speed_smoothness = speed_difference.square().mean()
        candidates = path.shape[1]
        if candidates < 2:
            zero = path.new_zeros(())
            return {
                "loss": zero, "distance": zero,
                "smoothness_loss": smoothness,
                "speed_smoothness_loss": speed_smoothness,
            }
        pairs = torch.triu_indices(candidates, candidates, 1, device=path.device)
        difference = coarse[:, pairs[0]] - coarse[:, pairs[1]]
        future_coarse = output["path_point_weight"][:, 0, ::4]
        squared = difference.square().sum(dim=-1)
        distance = (
            (squared * future_coarse[:, None]).sum(dim=-1)
            / future_coarse[:, None].sum(dim=-1).clamp(min=1.0)
        ).clamp(min=1e-8).sqrt()
        return {
            "loss": (margin - distance).clamp(min=0.0).square().mean(),
            "distance": distance.mean(),
            "smoothness_loss": smoothness,
            "speed_smoothness_loss": speed_smoothness,
        }


__all__ = ["StackPlannerActorCritic"]
