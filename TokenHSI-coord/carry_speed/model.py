"""Categorical speed-only PPO; spatial routes are observations, not actions."""

from dataclasses import asdict
from pathlib import Path
import os

import torch
from torch import nn
from torch.distributions import Categorical

from coordinator.geometry import state_to_tokens, world_to_shared
from stack_planner.model import StackPlannerConfig, StackSceneEncoder
from carry_speed.control import SPEED_CHOICES, controlled_mask


class CarrySpeedPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or StackPlannerConfig(history_steps=4, d_model=96, nhead=4,
                                                   feedforward=192, encoder_layers=2)
        self.encoder = StackSceneEncoder(self.config)
        self.actor = nn.Sequential(nn.Linear(self.config.d_model + 2, 96), nn.GELU(),
                                   nn.Linear(96, 2 * len(SPEED_CHOICES)))
        self.critic = nn.Sequential(nn.Linear(self.config.d_model + 2, 96), nn.GELU(), nn.Linear(96, 1))
        nn.init.normal_(self.actor[-1].weight, std=0.001)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, observation, priority):
        _, frame = state_to_tokens(observation.state)
        local = world_to_shared(observation.base_path_world, frame["center"], frame["angle"])
        scene = self.encoder(observation.history_tokens, observation.history_valid, local,
                             observation.base_path_valid, observation.path_progress)
        roles = torch.nn.functional.one_hot(priority, num_classes=2).to(scene.dtype)
        scene = torch.cat((scene, roles), dim=-1)
        return Categorical(logits=self.actor(scene).reshape(-1, 2, len(SPEED_CHOICES))), self.critic(scene).squeeze(-1)

    def evaluate(self, observation, priority, action):
        distribution, value = self.distribution(observation, priority)
        mask = controlled_mask(priority).to(value.dtype)
        return ((distribution.log_prob(action) * mask).sum(-1),
                (distribution.entropy() * mask).sum(-1), value)

    def act(self, observation, priority, deterministic=False):
        distribution, value = self.distribution(observation, priority)
        action = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        mask = controlled_mask(priority).to(value.dtype)
        return action, (distribution.log_prob(action) * mask).sum(-1), value

    def save(self, path, optimizer, step, executor, metrics):
        path = Path(path)
        payload = {"kind": "carry_fixed_route_speed_v1", "config": asdict(self.config),
                   "speed_choices": SPEED_CHOICES, "state": self.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step,
                   "executor": str(Path(executor).resolve()), "metrics": metrics}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device):
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("kind") != "carry_fixed_route_speed_v1" or tuple(payload.get("speed_choices", ())) != SPEED_CHOICES:
            raise ValueError("checkpoint is not a compatible fixed-route speed policy")
        model = cls(StackPlannerConfig(**payload["config"])).to(device)
        model.load_state_dict(payload["state"], strict=True)
        return model, payload
