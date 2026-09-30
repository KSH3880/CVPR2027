"""Online viewer for deterministic or sampled plain-Carry proposals."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from carry_planner.env_adapter import HumanoidMACarryPlannerTrain
from carry_planner.view_debug import (
    rejected_path_vertices, rejection_reason, viewer_cross_slots,
)
from stack_planner.checkpoint import load_stack_checkpoint
from stack_planner.history import StackHistoryBuffer
from stack_planner.policy import StackPlannerActorCritic


class HumanoidMACarryPlannerView(HumanoidMACarryPlannerTrain):
    """Replan online while the frozen two-agent Carry policy executes."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._carry_planner_view_ready = False
        self._carry_view_mixed_layout = bool(int(os.environ.get(
            "CARRY_PLANNER_VIEW_MIXED_LAYOUT", "0",
        )))
        self._carry_view_layout_count = 0
        self._carry_view_layout_jitter_m = float(os.environ.get(
            "CARRY_PLANNER_VIEW_LAYOUT_JITTER_M", "0",
        ))
        if not 0 <= self._carry_view_layout_jitter_m < float("inf"):
            raise ValueError(
                "CARRY_PLANNER_VIEW_LAYOUT_JITTER_M must be finite and nonnegative"
            )
        checkpoint = Path(
            os.environ.get("CARRY_PLANNER_CKPT", "")
        ).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "CARRY_PLANNER_CKPT must point to a planner checkpoint"
            )
        period = int(os.environ.get("CARRY_PLANNER_REPLAN_STEPS", "12"))
        if period < 1:
            raise ValueError("CARRY_PLANNER_REPLAN_STEPS must be positive")
        self._carry_planner_period = period
        self._carry_planner_stochastic = bool(int(os.environ.get(
            "CARRY_PLANNER_VIEW_STOCHASTIC", "0",
        )))
        self._carry_planner_goal_freeze_m = float(os.environ.get(
            "CARRY_PLANNER_VIEW_GOAL_FREEZE_M", "0",
        ))
        if not 0 <= self._carry_planner_goal_freeze_m < float("inf"):
            raise ValueError(
                "CARRY_PLANNER_VIEW_GOAL_FREEZE_M must be finite and nonnegative"
            )
        self._carry_planner_draw_rejected = bool(int(os.environ.get(
            "CARRY_PLANNER_DRAW_REJECTED", "1",
        )))
        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless,
        )
        self._carry_planner, payload = load_stack_checkpoint(
            checkpoint, self.device,
        )
        if not self._carry_planner.config.plain_carry:
            raise ValueError("checkpoint is not a plain-carry planner")
        if self._carry_planner.config.candidates != 1:
            raise ValueError("plain-carry viewer currently requires candidates=1")
        self._carry_planner.eval().requires_grad_(False)
        self._carry_planner_policy = None
        self._carry_planner_std_source = "none"
        if self._carry_planner_stochastic:
            self._carry_planner_policy = StackPlannerActorCritic(
                self._carry_planner,
                point_std=float(os.environ.get(
                    "CARRY_PLANNER_VIEW_DELTA_STD", "0.10",
                )),
                endpoint_std=0.03,
                anchor_std=0.03,
                speed_std=float(os.environ.get(
                    "CARRY_PLANNER_VIEW_SPEED_STD", "0.20",
                )),
            ).to(self.device)
            stored_std = payload.get("extras", {}).get("action_log_std")
            if stored_std is not None:
                self._carry_planner_policy.action_log_std.data.copy_(
                    stored_std.to(self.device)
                )
                self._carry_planner_std_source = "checkpoint"
            else:
                self._carry_planner_std_source = "viewer_env"
            self._carry_planner_policy.eval().requires_grad_(False)
        self._carry_planner_history = StackHistoryBuffer(
            self.num_envs, self._carry_planner.config.history_steps,
            self.device,
        )
        self._carry_planner_tick = torch.full(
            (self.num_envs,), -period, dtype=torch.long, device=self.device,
        )
        self._carry_planner_phase = torch.full(
            (self.num_envs, 2), -99.0, device=self.device,
        )
        self._carry_planner_replans = 0
        self._carry_planner_invalid = 0
        self._carry_planner_raw_path = None
        self._carry_planner_raw_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._carry_planner_goal_freeze = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device,
        )
        self._carry_view_install_freeze = None
        self._carry_planner_raw_frozen = torch.zeros_like(
            self._carry_planner_goal_freeze,
        )
        self._carry_planner_raw_valid = torch.ones_like(
            self._carry_planner_raw_ready,
        )
        self._carry_planner_view_ready = True
        self._compute_observations()
        mode = "stochastic" if self._carry_planner_stochastic else "deterministic"
        std = (
            float(self._carry_planner_policy.action_log_std[0, 0].exp())
            if self._carry_planner_policy is not None else 0.0
        )
        print(
            "[carry-planner-view] checkpoint={} schema={} step={} "
            "replan={} proposal_mode={} delta_std={} std_source={} "
            "frozen_agent=True goal_freeze_m={}".format(
                checkpoint.resolve(), payload["schema_version"],
                payload.get("step", 0), period, mode, std,
                self._carry_planner_std_source, self._carry_planner_goal_freeze_m,
            ),
            flush=True,
        )
        print(
            "[carry-planner-view] inherited ribbons show installed paths; "
            "rejected raw proposals are red; speed colors follow the "
            "commands sent to ms18",
            flush=True,
        )

    def _jitter_view_packages(self, env_ids):
        """Move each root, box and support together without breaking grasp pose."""
        if len(env_ids) == 0 or self._carry_view_layout_jitter_m == 0:
            return
        offset = (
            2.0 * torch.rand(len(env_ids), 2, 2, device=self.device) - 1.0
        ) * self._carry_view_layout_jitter_m
        self.agent_axis(self._humanoid_root_states)[env_ids, :, :2] += offset
        self.agent_axis(self._box_states)[env_ids, :, :2] += offset
        if self._carry_reset_random_height:
            self.agent_axis(self._platform_states)[env_ids, :, :2] += offset

    def apply_layout(self, env_ids):
        """Three randomized convergence scenes, then one timed Cross scene."""
        if (
            not self._carry_view_mixed_layout
            or getattr(self, "_view_timed_cross", False)
            or os.environ.get("MS_SCEN", "cross") != "cross"
            or self._carry_converge_prob != 0.75
        ):
            super().apply_layout(env_ids)
            if not getattr(self, "_view_timed_cross", False):
                self._jitter_view_packages(env_ids)
            return
        if len(env_ids) == 0:
            return
        cross = viewer_cross_slots(
            self._carry_view_layout_count, len(env_ids), env_ids.device,
        )
        self._carry_view_layout_count += len(env_ids)
        old_prob = self._carry_converge_prob
        try:
            self._carry_converge_prob = 0.0
            if bool(cross.any()):
                super().apply_layout(env_ids[cross])
            self._carry_converge_prob = 1.0
            if bool((~cross).any()):
                super().apply_layout(env_ids[~cross])
        finally:
            self._carry_converge_prob = old_prob
        self._jitter_view_packages(env_ids)
        if bool(cross.any()):
            old_cross_prob = self._view_timed_cross_prob
            try:
                self._view_timed_cross_prob = 1.0
                self._apply_view_timed_cross(env_ids[cross])
            finally:
                self._view_timed_cross_prob = old_cross_prob

    def _install_plan(self, env_ids, path, speed):
        """Keep the installed dense path for agents frozen by the viewer."""
        freeze = getattr(self, "_carry_view_install_freeze", None)
        if freeze is None or not bool(freeze[env_ids].any()):
            return super()._install_plan(env_ids, path, speed)
        rows = self.agent_rows(env_ids).reshape(-1, 2)[freeze[env_ids]]
        fields = (
            "_gt_path", "_mscale", "_s_end", "_arc_root", "_arc_box",
            "_prev_arc", "_coord_cmd_tick",
        )
        saved = {name: getattr(self, name)[rows].clone() for name in fields}
        super()._install_plan(env_ids, path, speed)
        for name, value in saved.items():
            getattr(self, name)[rows] = value

    @torch.no_grad()
    def _maybe_replan(self, env_ids):
        if not getattr(self, "_carry_planner_view_ready", False):
            return
        state = self.planner_state()
        restarted = self.progress_buf < self._carry_planner_tick
        if restarted.any():
            self._carry_planner_history.reset(restarted)
        due = (
            (self.progress_buf - self._carry_planner_tick)
            >= self._carry_planner_period
        )
        phase_changed = (state.phase != self._carry_planner_phase).any(dim=1)
        phase_changed_agent = (state.phase != self._carry_planner_phase)
        phase_changed_agent |= restarted[:, None]
        held = state.held >= 0.5
        eligible = (
            held & self._carry_planner_history.previous_path_valid[:, None]
            & ~phase_changed_agent
        )
        if (
            self._carry_planner_goal_freeze_m > 0
            and self._carry_planner.config.carry_suffix_replan
        ):
            near_goal = (
                (state.root_xy - state.goal_xy).norm(dim=-1)
                <= self._carry_planner_goal_freeze_m
            )
            self._carry_planner_goal_freeze = (
                self._carry_planner_goal_freeze | near_goal
            ) & eligible
        else:
            self._carry_planner_goal_freeze.zero_()
        selected = (
            (due | phase_changed | restarted)
            & ~self._carry_planner_goal_freeze.all(dim=1)
        )
        if not bool(selected.any()):
            return

        # Evaluation envs terminate asynchronously. A reset in one env must
        # not shift every other env's history or turn their 6-step period into
        # effectively every-step replanning.
        old_tokens = self._carry_planner_history.tokens.clone()
        old_valid = self._carry_planner_history.valid.clone()
        observation_all = self._carry_planner_history.observe(state, commit=True)
        self._carry_planner_history.tokens[~selected] = old_tokens[~selected]
        self._carry_planner_history.valid[~selected] = old_valid[~selected]
        observation = observation_all.index(selected)
        if self._carry_planner_policy is None:
            output = self._carry_planner(observation)
            history_path = output["path_world"][:, 0]
        else:
            output, _, _, _ = self._carry_planner_policy.sample_all(observation)
            # Match training: execute the sampled proposal, but keep exploration
            # noise out of the recurrent path reference.
            history_path = output["mean_path_world"][:, 0]
        frozen = self._carry_planner_goal_freeze[selected]
        proposed = output["path_world"][:, 0].clone()
        if bool(frozen.any()):
            output = dict(output)
            path = output["path_world"].clone()
            speed = output["speed"].clone()
            path[:, 0][frozen] = observation.previous_path_world[frozen]
            speed[:, 0][frozen] = 1.0
            output["path_world"] = path
            output["speed"] = speed
            history_path = history_path.clone()
            history_path[frozen] = observation.previous_path_world[frozen]
        env_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        self._carry_view_install_freeze = self._carry_planner_goal_freeze
        try:
            valid = self.install_external_plan(
                output, env_ids=env_ids, ignored_agents=frozen,
            )
        finally:
            self._carry_view_install_freeze = None
        if self._carry_planner_raw_path is None:
            self._carry_planner_raw_path = torch.zeros(
                (self.num_envs,) + tuple(proposed.shape[1:]),
                dtype=proposed.dtype, device=proposed.device,
            )
        self._carry_planner_raw_path[env_ids] = proposed
        self._carry_planner_raw_ready[env_ids] = True
        self._carry_planner_raw_valid[env_ids] = valid
        self._carry_planner_raw_frozen[env_ids] = frozen
        committed = self._carry_planner_history.previous_path_world.clone()
        committed[selected] = history_path
        base = self._carry_planner_history.base_path_world.clone()
        base[selected] = output["base_path_world"]
        update = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        update[selected] = valid
        self._carry_planner_history.commit_path(
            committed, update_mask=update, base_path_world=base,
        )
        self._carry_planner_tick[selected] = self.progress_buf[selected]
        self._carry_planner_phase[selected] = state.phase[selected]
        self._carry_planner_replans += len(env_ids)
        self._carry_planner_invalid += int((~valid).sum())
        if int(os.environ.get("CARRY_PLANNER_DEBUG", "1")):
            shown = int(env_ids[0])
            shown_local = 0
            diagnostics = self.last_plan_validity_debug()
            reason = (
                "accepted" if bool(valid[shown_local])
                else rejection_reason(diagnostics, shown_local)
            )
            print(
                "[carry-planner-view] env={} step={} phase={} frozen={} valid={} "
                "reason={} max_turn_deg={:.2f} cursor={} root={} box={} "
                "goal={}".format(
                    shown, int(self.progress_buf[shown]),
                    state.phase[shown].detach().cpu().tolist(),
                    frozen[shown_local].detach().cpu().tolist(),
                    bool(valid[shown_local]),
                    reason, float(diagnostics["max_turn_deg"][shown_local]),
                    self._arc_root.reshape(self.num_envs, 2)[shown]
                    .detach().cpu().tolist(),
                    state.root_xy[shown].detach().cpu().tolist(),
                    state.box_xyz[shown, :, :2].detach().cpu().tolist(),
                    state.goal_xy[shown].detach().cpu().tolist(),
                ),
                flush=True,
            )

    def _draw_task(self):
        super()._draw_task()
        if (
            not getattr(self, "_carry_planner_view_ready", False)
            or not self._carry_planner_draw_rejected
            or self.viewer is None
            or self._carry_planner_raw_path is None
            or not bool(self._carry_planner_raw_ready[0])
            or bool(self._carry_planner_raw_valid[0])
        ):
            return
        path = self._carry_planner_raw_path[0].detach().cpu().numpy()
        vertices = rejected_path_vertices(path)
        colors = np.asarray(
            ((1.0, 0.05, 0.05), (1.0, 0.30, 0.05)), dtype=np.float32,
        )
        for agent in range(vertices.shape[0]):
            if bool(self._carry_planner_raw_frozen[0, agent]):
                continue
            segment_count = vertices.shape[1]
            line_colors = np.repeat(
                colors[agent:agent + 1], segment_count, axis=0,
            )
            self.gym.add_lines(
                self.viewer, self.envs[0], segment_count,
                vertices[agent], line_colors,
            )

    def _reset_envs(self, env_ids):
        ready = getattr(self, "_carry_planner_view_ready", False)
        self._carry_planner_view_ready = False
        try:
            super()._reset_envs(env_ids)
        finally:
            self._carry_planner_view_ready = ready
        if ready and len(env_ids):
            mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device,
            )
            mask[env_ids] = True
            self._carry_planner_history.reset(mask)
            self._carry_planner_tick[env_ids] = -self._carry_planner_period
            self._carry_planner_phase[env_ids] = -99.0
            self._carry_planner_goal_freeze[env_ids] = False
            self._carry_planner_raw_ready[env_ids] = False
            self._compute_observations(env_ids)

    def report_metrics(self):
        super().report_metrics()
        if getattr(self, "_carry_planner_view_ready", False):
            print(
                "CARRY_PLANNER_VIEW_SUMMARY replans={} invalid={}".format(
                    self._carry_planner_replans,
                    self._carry_planner_invalid,
                ),
                flush=True,
            )


__all__ = ["HumanoidMACarryPlannerView"]
