"""Deterministic runtime task for viewing a trained stack planner."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .checkpoint import load_stack_checkpoint
from .env_adapter import HumanoidMAStackPlannerTrain


class HumanoidMAStackPlannerView(HumanoidMAStackPlannerTrain):
    """Run planner means online while the frozen stack policy executes them."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._stack_planner_view_ready = False
        checkpoint = Path(os.environ.get("STACK_PLANNER_CKPT", "")).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "STACK_PLANNER_CKPT must point to a stack planner checkpoint"
            )
        period = int(os.environ.get("STACK_PLANNER_REPLAN_STEPS", "6"))
        if period < 1:
            raise ValueError("STACK_PLANNER_REPLAN_STEPS must be positive")
        self._stack_planner_period = period
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._stack_planner, payload = load_stack_checkpoint(checkpoint, self.device)
        self._stack_planner.eval().requires_grad_(False)
        self._stack_planner_tick = torch.full(
            (self.num_envs,), -period, dtype=torch.long, device=self.device
        )
        self._stack_planner_last_phase = torch.full(
            (self.num_envs,), -99, dtype=torch.long, device=self.device
        )
        self._stack_planner_replans = 0
        self._stack_planner_invalid = 0
        self._stack_planner_unsafe = 0
        self._stack_planner_view_ready = True
        self._compute_observations()
        print(
            "[stack-planner-view] checkpoint={} schema={} step={} replan={} "
            "candidate=0 deterministic=True frozen_agent=True".format(
                checkpoint.resolve(), payload["schema_version"],
                payload.get("step", 0), period,
            ),
            flush=True,
        )

    @torch.no_grad()
    def _view_plan(self):
        phase = self._stack_phase
        due = (
            (self.progress_buf - self._stack_planner_tick
             >= self._stack_planner_period)
            | (phase != self._stack_planner_last_phase)
        )
        if not bool(due.any()):
            return
        # Viewer batches are intentionally small. Replanning the whole batch
        # keeps the install ABI simple and makes phase changes immediately visible.
        output = self._stack_planner(self.planner_state())
        valid, safe = self.install_external_plan(output)
        self._stack_planner_replans += self.num_envs
        self._stack_planner_invalid += int((~valid).sum())
        self._stack_planner_unsafe += int((valid & ~safe).sum())
        self._stack_planner_tick.copy_(self.progress_buf)
        self._stack_planner_last_phase.copy_(phase)

    def _compute_task_obs(self, env_ids=None):
        if getattr(self, "_stack_planner_view_ready", False):
            self._view_plan()
        return super()._compute_task_obs(env_ids)

    def _reset_envs(self, env_ids):
        ready = getattr(self, "_stack_planner_view_ready", False)
        self._stack_planner_view_ready = False
        try:
            super()._reset_envs(env_ids)
        finally:
            self._stack_planner_view_ready = ready
        if ready and len(env_ids):
            self._stack_planner_tick[env_ids] = -self._stack_planner_period
            self._stack_planner_last_phase[env_ids] = -99
            self._compute_observations(env_ids)

    def report_metrics(self):
        super().report_metrics()
        if getattr(self, "_stack_planner_view_ready", False):
            print(
                "STACK_PLANNER_VIEW_SUMMARY replans={} invalid={} unsafe={}".format(
                    self._stack_planner_replans,
                    self._stack_planner_invalid,
                    self._stack_planner_unsafe,
                ),
                flush=True,
            )


__all__ = ["HumanoidMAStackPlannerView"]
