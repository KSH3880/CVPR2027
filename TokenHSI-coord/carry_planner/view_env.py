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
        selected = due | phase_changed | restarted
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
        output = self._carry_planner(observation)
        env_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        valid = self.install_external_plan(output, env_ids=env_ids)
        committed = self._carry_planner_history.previous_path_world.clone()
        committed[selected] = output["path_world"][:, 0]
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
            print(
                "[carry-planner-view] env={} step={} phase={} valid={} "
                "cursor={} root={} box={} goal={}".format(
                    shown, int(self.progress_buf[shown]),
                    state.phase[shown].detach().cpu().tolist(),
                    bool(valid[shown_local]),
                    self._arc_root.reshape(self.num_envs, 2)[shown]
                    .detach().cpu().tolist(),
                    state.root_xy[shown].detach().cpu().tolist(),
                    state.box_xyz[shown, :, :2].detach().cpu().tolist(),
                    state.goal_xy[shown].detach().cpu().tolist(),
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
