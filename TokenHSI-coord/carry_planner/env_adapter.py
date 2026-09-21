"""Bridge the stack-planner path ABI to simultaneous two-agent Carry.

The neural planner, action distribution, history observation and checkpoint
format stay in :mod:`stack_planner`.  This adapter deliberately contains no
sequential-stack phase, handoff, retreat, or virtual-box behavior.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from carry_planner.layout import converging_goal_xy
from coordinator.schema import AGENTS
from env.tasks.adapt_interaction_skills.humanoid_ma_coord_carry import (
    HumanoidMACoordCarry,
)
from stack_planner.reset_transaction import CarryOnlySingleCommitReset


class HumanoidMACarryPlannerTrain(
    CarryOnlySingleCommitReset, HumanoidMACoordCarry,
):
    """Simultaneous Carry task controlled by an external joint planner."""

    def create_sim(self):
        """Keep CUDA compute logical while selecting Vulkan by physical id.

        With one UUID in CUDA_VISIBLE_DEVICES, Torch and PhysX must both use
        logical cuda:0. Isaac Gym's viewer graphics index is instead a Vulkan
        physical ordinal and ignores CUDA visibility, so only that index is
        replaced with the requested nvitop/NVML device.
        """
        raw = os.environ.get("CARRY_PLANNER_PHYSICAL_GPU")
        if raw is None:
            return super().create_sim()
        try:
            physical = int(raw)
        except ValueError as error:
            raise ValueError(
                "CARRY_PLANNER_PHYSICAL_GPU must be a non-negative integer"
            ) from error
        if physical < 0:
            raise ValueError(
                "CARRY_PLANNER_PHYSICAL_GPU must be a non-negative integer"
            )
        graphics = self.graphics_device_id
        if graphics >= 0:
            self.graphics_device_id = physical
        print(
            f"[carry-planner-device] torch_physx=cuda:{self.device_id} "
            f"visible_physical_gpu={physical} "
            f"graphics_physical_gpu={physical if graphics >= 0 else -1}",
            flush=True,
        )
        try:
            return super().create_sim()
        finally:
            self.graphics_device_id = graphics

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        provider = os.environ.get("COORD_PROVIDER", "external").lower()
        if provider != "external":
            raise ValueError("carry-planner training requires COORD_PROVIDER=external")
        self._carry_converge_prob = float(os.environ.get(
            "CARRY_PLANNER_CONVERGE_PROB", "0.75",
        ))
        self._carry_goal_margin = float(os.environ.get(
            "CARRY_PLANNER_GOAL_MARGIN", "0.25",
        ))
        if not 0.0 <= self._carry_converge_prob <= 1.0:
            raise ValueError("CARRY_PLANNER_CONVERGE_PROB must be in [0, 1]")
        if self._carry_goal_margin < 0.0:
            raise ValueError("CARRY_PLANNER_GOAL_MARGIN must be non-negative")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self.num_agents != AGENTS:
            raise ValueError(f"carry planner requires exactly {AGENTS} agents")

    def apply_layout(self, env_ids):
        """Mix ordinary Cross with feasible close-goal convergence cases."""
        super().apply_layout(env_ids)
        if not hasattr(self, "_carry_converge_layout"):
            self._carry_converge_layout = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device,
            )
        self._carry_converge_layout[env_ids] = False
        if len(env_ids) == 0 or self._carry_converge_prob <= 0.0:
            return
        selected = torch.rand(len(env_ids), device=self.device) < self._carry_converge_prob
        if not bool(selected.any()):
            return
        ids = env_ids[selected]
        rows = self.agent_rows(ids)
        initial = self.agent_axis(self._initial_humanoid_root_states)
        center = initial[ids, :, :2].mean(dim=1)
        size = self._box_lib._box_size[rows, :2].reshape(-1, AGENTS, 2)
        angle = torch.rand(len(ids), device=self.device) * (2.0 * torch.pi)
        targets = converging_goal_xy(
            center, size, self._carry_goal_margin, angle,
        )
        self._box_tar_pos[rows, :2] = targets.reshape(-1, 2)
        if self._carry_reset_random_height:
            self.agent_axis(self._tar_platform_states)[ids, :, :2] = targets

        self._carry_converge_layout[ids] = True

    def planner_state(self, env_ids=None):
        """Return only measured simulator state using the common planner ABI."""
        return self.coord_state(env_ids)

    def _install_plan(self, env_ids, path, speed):
        """Replace the trajectory without rewinding the executor cursor.

        ``HumanoidMACoordCarry`` normally receives a freshly rooted path on
        every decision, so its installer resets ``_arc_root`` to zero.  The
        recurrent Carry planner instead keeps an episode-fixed path prefix.
        Resetting that cursor at pickup therefore makes the steer window point
        back toward the episode start.  Preserve all path-relative cursors
        after the first accepted plan; only the trajectory and speed profile
        are replaced by the parent installer.
        """
        had_plan = self._coord_has_valid[env_ids].clone()
        previous_root = self._arc_root[self.agent_rows(env_ids)].clone()
        previous_box = self._arc_box[self.agent_rows(env_ids)].clone()
        previous_metric = self._prev_arc[self.agent_rows(env_ids)].clone()

        super()._install_plan(env_ids, path, speed)

        if had_plan.any():
            chosen_envs = env_ids[had_plan]
            chosen_rows = self.agent_rows(chosen_envs)
            chosen = had_plan.repeat_interleave(AGENTS)
            end = self._s_end[chosen_rows]
            self._arc_root[chosen_rows] = torch.minimum(
                previous_root[chosen], end,
            )
            self._arc_box[chosen_rows] = torch.minimum(
                previous_box[chosen], end,
            )
            self._prev_arc[chosen_rows] = torch.minimum(
                previous_metric[chosen], end,
            )

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
        # The shared stack planner has a fixed-origin recurrent trajectory:
        # executed prefix points remain where they were first committed while
        # the current root (and a held box) advances along that trajectory.
        # The older coordinator instead regenerates root/box anchors on every
        # replan. Its dynamic equality check is therefore inapplicable here.
        # Initial pickup/goal anchors are still hard-coded and action-masked by
        # StackTrajectoryPlanner; retain finite/buffer/speed/curvature checks.
        checks[:, 1] = True
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
