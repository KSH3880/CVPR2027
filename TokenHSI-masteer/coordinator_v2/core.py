"""Proposal-guided MPPI over an executor-aware recurrent world model.

The planner acts only in the high-level path/speed space consumed by MS18.
It never predicts or changes TokenHSI joint actions.  A plan is a short sequence
of per-agent ``(shared-frame dx, dy, speed-logit)`` knots plus pickup dwell.
The codec expands it to the stable 33-point root->box->goal contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from coordinator.geometry import rotate_xy, state_to_tokens
from coordinator.planner import candidate_costs
from coordinator.schema import AGENTS, MAX_SPEED, MIN_SPEED, PATH_POINTS, CoordinatorState


STATE_DIM = AGENTS * 3 * 12


def encode_state(state: CoordinatorState) -> torch.Tensor:
    """Encode refreshed simulator state using the existing normalized tokens."""
    tokens, _ = state_to_tokens(state)
    return tokens.flatten(start_dim=1)


@dataclass(frozen=True)
class PlannerConfig:
    history_frames: int = 8
    plan_horizon: int = 8
    hidden_dim: int = 128
    ensemble_size: int = 5
    task_dim: int = 0
    scene_dim: int = 0
    mppi_candidates: int = 32
    mppi_iterations: int = 3
    mppi_temperature: float = 1.0
    elite_std_min: float = 0.04
    elite_std_max: float = 1.0
    path_residual_scale: float = 1.0
    uncertainty_weight: float = 1.0
    world_return_weight: float = 1.0
    safety_penalty: float = 1.0e6
    result_candidates: int = 4

    def __post_init__(self):
        if self.plan_horizon < 4 or self.plan_horizon % 2:
            raise ValueError("plan_horizon must be an even integer >= 4")
        for name in ("history_frames", "hidden_dim", "ensemble_size", "mppi_candidates", "mppi_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.result_candidates <= min(4, self.mppi_candidates):
            raise ValueError("result_candidates must be in [1, min(4, mppi_candidates)]")

    def as_dict(self):
        return asdict(self)


class HistoryEncoder(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.input_dim = STATE_DIM + config.task_dim + config.scene_dim
        self.gru = nn.GRU(self.input_dim, config.hidden_dim, batch_first=True)

    def forward(self, history, task=None, scene=None):
        if history.ndim != 3 or history.shape[-1] != STATE_DIM:
            raise ValueError(f"history must be [B,K,{STATE_DIM}]")
        parts = [history]
        batch, frames = history.shape[:2]
        if task is not None:
            parts.append(task[:, None].expand(batch, frames, -1))
        if scene is not None:
            parts.append(scene[:, None].expand(batch, frames, -1))
        value = torch.cat(parts, dim=-1)
        if value.shape[-1] != self.input_dim:
            raise ValueError(f"history context width {value.shape[-1]} != {self.input_dim}")
        _, hidden = self.gru(value)
        return hidden[-1]


class PlanProposal(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        width = config.plan_horizon * AGENTS * 3
        self.body = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, width)
        self.dwell = nn.Linear(config.hidden_dim, AGENTS)
        self.log_std = nn.Parameter(torch.full((config.plan_horizon, AGENTS, 3), -1.6))
        self.dwell_log_std = nn.Parameter(torch.full((AGENTS,), -1.6))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            self.mean.bias.view(config.plan_horizon, AGENTS, 3)[..., 2] = -4.0
        nn.init.zeros_(self.dwell.weight)
        nn.init.constant_(self.dwell.bias, math.log(math.expm1(1.5)))

    def forward(self, context):
        hidden = self.body(context)
        mean = self.mean(hidden).reshape(-1, self.config.plan_horizon, AGENTS, 3)
        dwell = self.dwell(hidden)
        return {
            "mean": mean,
            "std": self.log_std.exp().expand_as(mean),
            "dwell_mean": dwell,
            "dwell_std": self.dwell_log_std.exp().expand_as(dwell),
        }

    def alignment_loss(self, context, action, dwell_raw, value, temperature=1.0):
        proposal = self(context)
        std = proposal["std"].clamp_min(1e-5)
        dstd = proposal["dwell_std"].clamp_min(1e-5)
        nll = 0.5 * (((action - proposal["mean"]) / std).square() + 2.0 * std.log())
        nll = nll.flatten(start_dim=1).mean(dim=1)
        nll += 0.5 * (((dwell_raw - proposal["dwell_mean"]) / dstd).square() + 2.0 * dstd.log()).mean(dim=1)
        weight = torch.softmax(value.detach() / float(temperature), dim=0)
        return (weight * nll).sum()


class PlanCodec(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def _interp_controls(value, length):
        shape = value.shape
        flat = value.movedim(-2, -1).reshape(-1, shape[-1], shape[-2])
        out = F.interpolate(flat, size=length, mode="linear", align_corners=True)
        return out.reshape(*shape[:-2], shape[-1], length).movedim(-1, -2)

    def forward(self, action, dwell_raw, state: CoordinatorState):
        if action.ndim != 5 or action.shape[2:] != (self.config.plan_horizon, AGENTS, 3):
            raise ValueError("action must be [B,K,H,2,3]")
        batch, candidates = action.shape[:2]
        root = state.root_xy[:, None]
        box = state.box_xyz[..., :2][:, None]
        goal = state.goal_xy[:, None]
        t = torch.linspace(0, 1, 17, device=action.device, dtype=action.dtype).reshape(1, 1, 1, 17, 1)
        first = root[..., None, :] + t * (box - root)[..., None, :]
        second = box[..., None, :] + t * (goal - box)[..., None, :]
        base = torch.cat((first, second[..., 1:, :]), dim=-2).expand(-1, candidates, -1, -1, -1).clone()

        half = self.config.plan_horizon // 2
        residual = action[..., :2].permute(0, 1, 3, 2, 4)
        legs = []
        for controls in (residual[..., :half, :], residual[..., half:, :]):
            zero = torch.zeros_like(controls[..., :1, :])
            legs.append(self._interp_controls(torch.cat((zero, controls, zero), dim=-2), 17)[..., 1:-1, :])
        residual33 = torch.cat((torch.zeros_like(base[..., :1, :]), legs[0],
                                torch.zeros_like(base[..., :1, :]), legs[1],
                                torch.zeros_like(base[..., :1, :])), dim=-2)
        angle = state.heading[:, 0][:, None, None, None]
        residual_world = rotate_xy(residual33, angle)
        path = base + self.config.path_residual_scale * torch.tanh(residual_world)
        path[..., 0, :] = root
        path[..., 16, :] = box
        path[..., 32, :] = goal

        speed_raw = action[..., 2].permute(0, 1, 3, 2).unsqueeze(-1)
        speed_raw = self._interp_controls(speed_raw, PATH_POINTS).squeeze(-1)
        speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * torch.sigmoid(speed_raw)
        dwell = F.softplus(dwell_raw)
        dwell = torch.where(state.held[:, None] >= 0.5, torch.zeros_like(dwell), dwell)
        return {"path_world": path, "speed": speed, "pickup_dwell": dwell}


class WorldModelMember(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.cell = nn.GRUCell(AGENTS * 3, config.hidden_dim)
        self.state_delta = nn.Linear(config.hidden_dim, STATE_DIM)
        self.reward = nn.Linear(config.hidden_dim, 1)
        self.collision = nn.Linear(config.hidden_dim, 1)
        self.success = nn.Linear(config.hidden_dim, 1)
        self.value = nn.Linear(config.hidden_dim, 1)
        for head in (self.state_delta, self.reward, self.collision, self.success, self.value):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def rollout(self, context, action):
        hidden = context
        rewards, states, collision, success = [], [], [], []
        for step in range(action.shape[1]):
            hidden = self.cell(action[:, step].flatten(start_dim=1), hidden)
            rewards.append(self.reward(hidden).squeeze(-1))
            states.append(self.state_delta(hidden))
            collision.append(torch.sigmoid(self.collision(hidden).squeeze(-1)))
            success.append(torch.sigmoid(self.success(hidden).squeeze(-1)))
        reward = torch.stack(rewards, dim=1).sum(dim=1) + self.value(hidden).squeeze(-1)
        collision_prob = 1.0 - torch.prod(1.0 - torch.stack(collision, dim=1), dim=1)
        return {
            "return": reward,
            "collision": collision_prob,
            "success": success[-1],
            "state_delta": torch.stack(states, dim=1),
            "reward_steps": torch.stack(rewards, dim=1),
            "collision_steps": torch.stack(collision, dim=1),
            "success_steps": torch.stack(success, dim=1),
        }


class EnsembleWorldModel(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.members = nn.ModuleList([WorldModelMember(config) for _ in range(config.ensemble_size)])

    def rollout(self, context, action):
        values = [member.rollout(context, action) for member in self.members]
        returns = torch.stack([x["return"] for x in values])
        return {
            "return": returns.mean(dim=0),
            "uncertainty": returns.var(dim=0, unbiased=False),
            "collision": torch.stack([x["collision"] for x in values]).mean(dim=0),
            "success": torch.stack([x["success"] for x in values]).mean(dim=0),
            "state_delta": torch.stack([x["state_delta"] for x in values]).mean(dim=0),
        }

    def supervised_loss(self, context, action, target, mask=None):
        if mask is None:
            mask = torch.ones(action.shape[:2], device=action.device, dtype=action.dtype)
        denom = mask.sum().clamp_min(1.0)
        losses = []
        for member in self.members:
            prediction = member.rollout(context, action)
            state = ((prediction["state_delta"] - target["state_delta"]).square().mean(dim=-1) * mask).sum() / denom
            reward = ((prediction["reward_steps"] - target["reward"]).square() * mask).sum() / denom
            collision = (F.binary_cross_entropy(prediction["collision_steps"], target["collision"], reduction="none") * mask).sum() / denom
            success = (F.binary_cross_entropy(prediction["success_steps"], target["success"], reduction="none") * mask).sum() / denom
            losses.append(state + reward + collision + success)
        return torch.stack(losses).mean()


@dataclass
class PlannerResult:
    path_world: torch.Tensor
    speed: torch.Tensor
    pickup_dwell: torch.Tensor
    valid: torch.Tensor
    safe: torch.Tensor
    index: torch.Tensor
    candidate_path_world: torch.Tensor
    candidate_cost: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]
    action: torch.Tensor
    dwell_raw: torch.Tensor


class WorldModelPlanner(nn.Module):
    def __init__(self, config: Optional[PlannerConfig] = None):
        super().__init__()
        self.config = config or PlannerConfig()
        self.encoder = HistoryEncoder(self.config)
        self.proposal = PlanProposal(self.config)
        self.codec = PlanCodec(self.config)
        self.world_model = EnsembleWorldModel(self.config)

    def encode_history(self, history, task=None, scene=None):
        return self.encoder(history, task, scene)

    def training_losses(self, history, action, target, *, planner_action=None,
                        planner_dwell=None, planner_value=None, mask=None,
                        task=None, scene=None):
        context = self.encode_history(history, task, scene)
        losses = {"world_model": self.world_model.supervised_loss(
            context, action, target, mask
        )}
        if planner_action is not None:
            if planner_dwell is None or planner_value is None:
                raise ValueError("planner_dwell and planner_value are required for alignment")
            losses["alignment"] = self.proposal.alignment_loss(
                context, planner_action, planner_dwell, planner_value
            )
        return losses

    def _evaluate(self, context, action, dwell_raw, state):
        batch, candidates = action.shape[:2]
        decoded = self.codec(action, dwell_raw, state)
        diagnostics = candidate_costs(decoded, state, measured_executor_timing=True)
        expanded = context[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        wm = self.world_model.rollout(expanded, action.reshape(batch * candidates, *action.shape[2:]))
        predicted = wm["return"].reshape(batch, candidates)
        uncertainty = wm["uncertainty"].reshape(batch, candidates)
        cost = diagnostics["train_cost"] - self.config.world_return_weight * predicted
        cost = cost + self.config.uncertainty_weight * uncertainty
        cost = cost + (~diagnostics["safe"]).to(cost.dtype) * self.config.safety_penalty
        cost = torch.where(diagnostics["valid"], cost, torch.full_like(cost, 1.0e9))
        return decoded, diagnostics, wm, cost

    @torch.no_grad()
    def plan(self, history, state: CoordinatorState, task=None, scene=None, generator=None):
        context = self.encode_history(history, task, scene)
        proposal = self.proposal(context)
        mean, std = proposal["mean"], proposal["std"]
        dmean, dstd = proposal["dwell_mean"], proposal["dwell_std"]
        population = self.config.mppi_candidates
        last = None
        for _ in range(self.config.mppi_iterations):
            noise = torch.randn((state.batch_size, population, *mean.shape[1:]), device=mean.device,
                                dtype=mean.dtype, generator=generator)
            dnoise = torch.randn((state.batch_size, population, AGENTS), device=mean.device,
                                 dtype=mean.dtype, generator=generator)
            action = mean[:, None] + std[:, None] * noise
            dwell = dmean[:, None] + dstd[:, None] * dnoise
            decoded, diagnostics, wm, cost = self._evaluate(context, action, dwell, state)
            weight = torch.softmax(-(cost - cost.amin(dim=1, keepdim=True)) / self.config.mppi_temperature, dim=1)
            mean = (weight[..., None, None, None] * action).sum(dim=1)
            dmean = (weight[..., None] * dwell).sum(dim=1)
            std = torch.sqrt((weight[..., None, None, None] * (action - mean[:, None]).square()).sum(dim=1) + 1e-6)
            dstd = torch.sqrt((weight[..., None] * (dwell - dmean[:, None]).square()).sum(dim=1) + 1e-6)
            std = std.clamp(self.config.elite_std_min, self.config.elite_std_max)
            dstd = dstd.clamp(self.config.elite_std_min, self.config.elite_std_max)
            last = (action, dwell, decoded, diagnostics, cost)
        action, dwell, decoded, diagnostics, cost = last
        index = cost.argmin(dim=1)
        row = torch.arange(state.batch_size, device=cost.device)
        top = cost.argsort(dim=1)[:, :self.config.result_candidates]
        gather_path = top[..., None, None, None].expand(-1, -1, AGENTS, PATH_POINTS, 2)
        candidates = decoded["path_world"].gather(1, gather_path)
        return PlannerResult(
            path_world=decoded["path_world"][row, index], speed=decoded["speed"][row, index],
            pickup_dwell=decoded["pickup_dwell"][row, index],
            valid=diagnostics["valid"][row, index], safe=diagnostics["safe"][row, index],
            index=index, candidate_path_world=candidates, candidate_cost=cost,
            diagnostics=diagnostics, action=action[row, index], dwell_raw=dwell[row, index],
        )

