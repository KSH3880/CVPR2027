"""Deterministic online viewer for a trained plain-Carry planner."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from carry_planner.env_adapter import HumanoidMACarryPlannerTrain
from stack_planner.checkpoint import load_stack_checkpoint
from stack_planner.history import StackHistoryBuffer


class HumanoidMACarryPlannerView(HumanoidMACarryPlannerTrain):
    """Replan online while the frozen two-agent Carry policy executes."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._carry_planner_view_ready = False
        checkpoint = Path(
            os.environ.get("CARRY_PLANNER_CKPT", "")
        ).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "CARRY_PLANNER_CKPT must point to a planner checkpoint"
            )
        period = int(os.environ.get("CARRY_PLANNER_REPLAN_STEPS", "6"))
        if period < 1:
            raise ValueError("CARRY_PLANNER_REPLAN_STEPS must be positive")
        self._carry_planner_period = period
        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless,
        )
        if self.num_envs != 1:
            raise ValueError("carry planner viewer currently requires exactly one env")

        self._carry_planner, payload = load_stack_checkpoint(
            checkpoint, self.device,
        )
        if not self._carry_planner.config.plain_carry:
            raise ValueError("checkpoint is not a plain-carry planner")
        if self._carry_planner.config.candidates != 1:
            raise ValueError("plain-carry viewer currently requires candidates=1")
        self._carry_planner.eval().requires_grad_(False)
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
        self._carry_planner_view_ready = True
        self._compute_observations()
        print(
            "[carry-planner-view] checkpoint={} schema={} step={} "
            "replan={} deterministic=True frozen_agent=True".format(
                checkpoint.resolve(), payload["schema_version"],
                payload.get("step", 0), period,
            ),
            flush=True,
        )
        print(
            "[carry-planner-view] inherited ribbons show installed paths; "
            "speed colors follow the commands sent to ms18",
            flush=True,
        )

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
        if not bool((due | phase_changed).any()):
            return

        observation = self._carry_planner_history.observe(state, commit=True)
        output = self._carry_planner(observation)
        valid = self.install_external_plan(output)
        self._carry_planner_history.commit_path(
            output["path_world"][:, 0], update_mask=valid,
            base_path_world=output["base_path_world"],
        )
        self._carry_planner_tick.copy_(self.progress_buf)
        self._carry_planner_phase.copy_(state.phase)
        self._carry_planner_replans += self.num_envs
        self._carry_planner_invalid += int((~valid).sum())
        if int(os.environ.get("CARRY_PLANNER_DEBUG", "1")):
            print(
                "[carry-planner-view] step={} phase={} valid={} "
                "cursor={} root={} box={} goal={}".format(
                    int(self.progress_buf[0]),
                    state.phase[0].detach().cpu().tolist(),
                    bool(valid[0]),
                    self._arc_root.reshape(self.num_envs, 2)[0]
                    .detach().cpu().tolist(),
                    state.root_xy[0].detach().cpu().tolist(),
                    state.box_xyz[0, :, :2].detach().cpu().tolist(),
                    state.goal_xy[0].detach().cpu().tolist(),
                ),
                flush=True,
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
