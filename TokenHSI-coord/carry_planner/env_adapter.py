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
from carry_planner.episode import initial_timeout_deadlines
from carry_planner.held_state import observed_box_held
from carry_planner.reward import carry_remaining_distance
from carry_planner.validity_debug import carry_plan_validity_debug
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
        # Stagger only the first timeout. Later episodes retain their normal
        # full length, and the task's progress counter always stays truthful.
        self._carry_timeout_deadline = initial_timeout_deadlines(
            self.num_envs, self.max_episode_length, self.device,
        )

    def _reset_envs(self, env_ids):
        # The player may call reset twice before the first simulated step.
        # Preserve the sampled first deadline across those empty resets.
        live = None
        if hasattr(self, "_carry_timeout_deadline") and len(env_ids) > 0:
            live = env_ids[self.progress_buf[env_ids] > 0]
        super()._reset_envs(env_ids)
        if live is not None and len(live) > 0:
            self._carry_timeout_deadline[env_ids] = self.max_episode_length

    def _compute_reset(self):
        super()._compute_reset()
        if not hasattr(self, "_carry_timeout_deadline"):
            return
        forced = (
            (self.progress_buf >= self._carry_timeout_deadline - 1)
            & ~self.reset_buf.bool()
        )
        if not bool(forced.any()):
            return
        # The parent has already booked ordinary terminal rewards. Apply the
        # makespan bonus only to these newly timed-out environments.
        original = self.reset_buf.clone()
        self.reset_buf[:] = forced.long()
        rows = self.agent_rows(torch.nonzero(forced, as_tuple=False).squeeze(-1))
        before = self.rew_buf[rows].clone()
        self._apply_makespan_reward()
        self._ep_task_r[rows] += self.rew_buf[rows] - before
        self.reset_buf[:] = original | forced.long()

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
        crossing = initial[ids, :, :2].mean(dim=1)
        box_xy = self.agent_axis(self._box_states)[ids, :, :2]
        size = self._box_lib._box_size[rows, :2].reshape(-1, AGENTS, 2)
        targets, feasible = converging_goal_xy(
            box_xy, crossing, size, self._carry_goal_margin,
        )
        if not bool(feasible.any()):
            return
        ids = ids[feasible]
        targets = targets[feasible]
        rows = self.agent_rows(ids)
        self._box_tar_pos[rows, :2] = targets.reshape(-1, 2)
        if self._carry_reset_random_height:
            self.agent_axis(self._tar_platform_states)[ids, :, :2] = targets

        self._carry_converge_layout[ids] = True

    def _coord_state(self, env_ids):
        """Use observed hand contact as well as height for planner pickup."""
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long,
            )
        state = super()._coord_state(env_ids)
        rows = self.agent_rows(env_ids)
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands = rigid[:, self._key_body_ids[[0, 1]]].mean(dim=1)
        hands = hands.reshape(len(env_ids), AGENTS, 3)
        box_size_z = self._box_lib._box_size[rows, 2].reshape(
            len(env_ids), AGENTS,
        )
        held = observed_box_held(
            state.root_xy, state.box_xyz, box_size_z, hands,
        )
        root_near = (
            (state.root_xy - state.box_xyz[..., :2]).norm(dim=-1) <= 0.7
        )
        goal_near = (
            (state.box_xyz[..., :2] - state.goal_xy).norm(dim=-1) <= 0.15
        )
        state.held = held.to(state.root_xy.dtype)
        state.phase = torch.where(
            goal_near & ~held, 3.0,
            torch.where(held, 2.0, root_near.to(state.root_xy.dtype)),
        )
        return state

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
        if getattr(self, "_carry_reset_cursor_on_install", False):
            super()._install_plan(env_ids, path, speed)
            return

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
    def install_external_plan(self, output, env_ids=None, ignored_agents=None):
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
        suffix_replan = bool(output.get("suffix_replan", False))
        box_index = output.get("box_index")
        if box_index is not None and box_index.ndim == 3:
            box_index = box_index[:, 0]
        state = self._coord_state(env_ids)
        lost_box = (
            (self._coord_phase[env_ids] == 2.0)
            & (state.held < 0.5)
        ).any(dim=1)
        if lost_box.any():
            self._coord_has_valid[env_ids[lost_box]] = False
        diagnostics = carry_plan_validity_debug(
            state, path, speed, box_index=box_index,
            suffix_replan=suffix_replan, ignored_agents=ignored_agents,
        )
        if suffix_replan:
            checks = torch.stack((
                diagnostics["finite"], diagnostics["anchors"],
                diagnostics["buffer"], diagnostics["speed"],
                diagnostics["curve"],
            ), dim=-1)
        else:
            checks = self._plan_validity_checks(state, path, speed)
            # Fixed-origin legacy Carry retains its episode-start point, so the
            # older dynamic anchor equality check is inapplicable.
            checks[:, 1] = True
        self._carry_last_plan_validity_debug = {
            key: value.detach() for key, value in diagnostics.items()
        }
        valid = checks.all(dim=-1)
        self._carry_reset_cursor_on_install = suffix_replan
        self._coord_replans[env_ids] += 1
        self._coord_invalid[env_ids] += (~valid).long()
        self._coord_invalid_reasons += (~checks).sum(dim=0)
        curve_invalid = ~checks[:, 4]
        held_count = (state.held >= 0.5).sum(dim=-1).long()
        for count in range(3):
            self._coord_curve_invalid_by_held[count] += (
                curve_invalid & (held_count == count)
            ).sum()
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

    def last_plan_validity_debug(self):
        # Diagnostics for the proposal checked most recently.
        if not hasattr(self, "_carry_last_plan_validity_debug"):
            raise RuntimeError("no Carry proposal has been checked yet")
        return self._carry_last_plan_validity_debug

    def planner_task_distance(self, state=None):
        """Continuous task distance for dense progress reward."""
        state = self.planner_state() if state is None else state
        return carry_remaining_distance(state)

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
