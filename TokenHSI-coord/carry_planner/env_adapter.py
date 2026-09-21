"""Bridge the stack-planner path ABI to simultaneous two-agent Carry.

The neural planner, action distribution, history observation and checkpoint
format stay in :mod:`stack_planner`.  This adapter deliberately contains no
sequential-stack phase, handoff, retreat, or virtual-box behavior.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from coordinator.schema import AGENTS
from env.tasks.adapt_interaction_skills.humanoid_ma_coord_carry import (
    HumanoidMACoordCarry,
)


class HumanoidMACarryPlannerTrain(HumanoidMACoordCarry):
    """Simultaneous Carry task controlled by an external joint planner."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        provider = os.environ.get("COORD_PROVIDER", "external").lower()
        if provider != "external":
            raise ValueError("carry-planner training requires COORD_PROVIDER=external")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self.num_agents != AGENTS:
            raise ValueError(f"carry planner requires exactly {AGENTS} agents")

    def planner_state(self, env_ids=None):
        """Return only measured simulator state using the common planner ABI."""
        return self.coord_state(env_ids)

    @torch.no_grad()
    def install_external_plan(self, output, env_ids=None):
        """Install one stack-planner proposal without coordinator selection.

        The default carry planner has one head.  Keeping this boundary strict
        avoids silently selecting candidates with the older coordinator's
        hand-written evaluator.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long,
            )
        path = output["path_world"]
        speed = output["speed"]
        if path.ndim != 5 or speed.ndim != 4 or path.shape[1] != 1:
            raise ValueError("carry planner expects one [B,1,2,33] proposal")
        path = path[:, 0]
        speed = speed[:, 0]
        state = self._coord_state(env_ids)
        checks = self._plan_validity_checks(state, path, speed)
        valid = checks.all(dim=-1)
        self._coord_replans[env_ids] += 1
        self._coord_invalid[env_ids] += (~valid).long()
        self._coord_invalid_reasons += (~checks).sum(dim=0)
        self._coord_selected[env_ids] = 0

        if valid.any():
            chosen = env_ids[valid]
            self._install_plan(chosen, path[valid], speed[valid])
            self._coord_has_valid[chosen] = True

        need_fallback = (~valid) & (~self._coord_has_valid[env_ids])
        if need_fallback.any():
            fallback_state = state.index(need_fallback)
            fallback_path, fallback_speed = self._analytic_plan(fallback_state)
            chosen = env_ids[need_fallback]
            self._install_plan(chosen, fallback_path, fallback_speed)
            self._coord_has_valid[chosen] = True
            self._coord_fallback[chosen] += 1
        self._coord_last_replan[env_ids] = self.progress_buf[env_ids]
        self._coord_phase[env_ids] = state.phase
        return valid

    def planner_task_distance(self, state=None):
        """Mean remaining approach/carry distance for dense progress reward."""
        state = self.planner_state() if state is None else state
        carrying = state.held >= 0.5
        source = torch.where(
            carrying[..., None], state.box_xyz[..., :2], state.root_xy,
        )
        target = torch.where(
            carrying[..., None], state.goal_xy, state.box_xyz[..., :2],
        )
        return (source - target).norm(dim=-1).mean(dim=1)

    def planner_collision_terms(self, state=None):
        """Differentiation-free physical proximity costs used after rollout."""
        state = self.planner_state() if state is None else state
        root = state.root_xy
        box = state.box_xyz[..., :2]
        radius = 0.5 * state.box_size_xy.norm(dim=-1)
        humanoid = F.relu(
            1.0 - (root[:, 0] - root[:, 1]).norm(dim=-1)
        ).square()
        box_limit = radius.sum(dim=-1) + 0.15
        boxes = F.relu(
            box_limit - (box[:, 0] - box[:, 1]).norm(dim=-1)
        ).square()
        human_box_01 = F.relu(
            0.35 + radius[:, 1]
            - (root[:, 0] - box[:, 1]).norm(dim=-1)
        ).square()
        human_box_10 = F.relu(
            0.35 + radius[:, 0]
            - (root[:, 1] - box[:, 0]).norm(dim=-1)
        ).square()
        human_box = 0.5 * (human_box_01 + human_box_10)
        return {
            "agent_agent": humanoid,
            "box_box": boxes,
            "agent_box": human_box,
            "total": humanoid + boxes + human_box,
        }


__all__ = ["HumanoidMACarryPlannerTrain"]
