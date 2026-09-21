"""Deterministic runtime task for viewing a trained stack planner."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import numpy as np

from .checkpoint import load_stack_checkpoint
from .constraints import free_path_validity_details
from .env_adapter import HumanoidMAStackPlannerTrain
from .history import StackHistoryBuffer


class HumanoidMAStackPlannerView(HumanoidMAStackPlannerTrain):
    """Run planner means online while the frozen stack policy executes them."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._stack_planner_view_ready = False
        checkpoint = Path(os.environ.get("STACK_PLANNER_CKPT", "")).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "STACK_PLANNER_CKPT must point to a stack planner checkpoint"
            )
        period = int(os.environ.get("STACK_PLANNER_REPLAN_STEPS", "30"))
        if period < 1:
            raise ValueError("STACK_PLANNER_REPLAN_STEPS must be positive")
        self._stack_planner_period = period
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._stack_planner, payload = load_stack_checkpoint(checkpoint, self.device)
        self._stack_planner.eval().requires_grad_(False)
        self._stack_history = StackHistoryBuffer(
            self.num_envs, self._stack_planner.config.history_steps,
            self.device,
        )
        self._stack_planner_tick = torch.full(
            (self.num_envs,), -period, dtype=torch.long, device=self.device
        )
        self._stack_planner_last_phase = torch.full(
            (self.num_envs,), -99, dtype=torch.long, device=self.device
        )
        self._stack_planner_replans = 0
        self._stack_planner_invalid = 0
        self._stack_planner_unsafe = 0
        self._stack_planner_latest_path = None
        self._stack_planner_status = None
        self._stack_planner_view_ready = True
        self._compute_observations()
        print(
            "[stack-planner-view] checkpoint={} schema={} step={} replan={} "
            "candidates={} learned_argmax=True deterministic=True frozen_agent=True".format(
                checkpoint.resolve(), payload["schema_version"],
                payload.get("step", 0), period,
                self._stack_planner.config.candidates,
            ),
            flush=True,
        )
        print("[stack-planner-view] white=latest raw model output; "
              "blue wire box=virtual retreat box", flush=True)

    def _update_marker(self):
        """Do not submit disabled multi-task actor IDs from the viewer.

        The inherited debug-path renderer calls this before drawing its line
        overlays.  Its generic implementation commits traj, sit, carry and
        climb actors together, but this carry-only stack scene intentionally
        has no valid sit/climb actors.  The task targets and boxes are already
        committed by reset/physics, so no per-render root-state write is
        required for the planner path visualization.
        """
        return

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
        restarted = self.progress_buf < self._stack_planner_tick
        if restarted.any():
            self._stack_history.reset(restarted)
        # Viewer batches are intentionally small. Replanning the whole batch
        # keeps the install ABI simple and makes phase changes immediately visible.
        state = self.planner_state()
        observation = self._stack_history.observe(state, commit=True)
        output = self._stack_planner(observation)
        selected = output["selected_candidate"]
        self._stack_planner_latest_path = output["path_world"][:, 0].detach().clone()
        valid, safe = self.install_external_plan(output)
        self._stack_history.commit_path(
            output["path_world"][:, 0],
            update_mask=valid & self._planner_policy_decision,
            base_path_world=output["base_path_world"],
        )
        status = (phase.cpu().tolist(), selected.cpu().tolist(), valid.cpu().tolist(),
                  self._planner_plan_installed.cpu().tolist(),
                  self._planner_retreat_ready.cpu().tolist())
        if int(os.environ.get("STACK_DEBUG", "0")) != 0:
            endpoint = output["path_world"][:, 0, 0, -1]
            virtual = self._planner_virtual_retreat_pos[:, :2]
            error = torch.norm(endpoint - virtual, dim=-1)
            validity = free_path_validity_details(
                output["path_world"], output["speed"], state.root_xy,
                self.planner_execution_rows(),
            )
            print(
                "[stack-planner-view] endpoint={} virtual={} error={} valid={}".format(
                    endpoint[0].detach().cpu().tolist(),
                    virtual[0].detach().cpu().tolist(),
                    float(error[0]), bool(valid[0]),
                ),
                flush=True,
            )
            print(
                "[stack-planner-view] validity agent={} finite={} root={} "
                "buffer={} speed={} max_turn_deg={} path_length={}".format(
                    validity["agent_valid"][0, 0].detach().cpu().tolist(),
                    validity["finite"][0, 0].detach().cpu().tolist(),
                    validity["root_reachable"][0, 0].detach().cpu().tolist(),
                    validity["buffer_ok"][0, 0].detach().cpu().tolist(),
                    validity["speed_ok"][0, 0].detach().cpu().tolist(),
                    validity["max_turn_deg"][0, 0].detach().cpu().tolist(),
                    validity["path_length"][0, 0].detach().cpu().tolist(),
                ),
                flush=True,
            )
            print(
                "[stack-planner-view] a2_gate stable_seen={} delay={}/{}".format(
                    bool(self._planner_bottom_stable_seen[0]),
                    int(self._planner_a2_delay_count[0]),
                    self._planner_a2_stable_delay_steps,
                ),
                flush=True,
            )
        if status != self._stack_planner_status:
            print("[stack-planner-view] phase={} candidate={} valid={} installed={} retreat_ready={}"
                  .format(*[item[:4] for item in status]), flush=True)
            self._stack_planner_status = status
        self._stack_planner_replans += self.num_envs
        self._stack_planner_invalid += int((~valid).sum())
        self._stack_planner_unsafe += int((valid & ~safe).sum())
        self._stack_planner_tick.copy_(self.progress_buf)
        self._stack_planner_last_phase.copy_(phase)

    def _draw_task(self):
        """Show only raw planner proposals and the virtual retreat box.

        Do not use the inherited speed-run renderer: its point decimation and
        run filtering can omit short curves and endpoints entirely.
        """
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        if getattr(self, "_stack_planner_latest_path", None) is None:
            return
        raw = self._stack_planner_latest_path.cpu().numpy()
        virtual = self._planner_virtual_retreat_pos.cpu().numpy()
        sizes = self._box_lib._box_size.cpu().numpy()
        phase = self._stack_phase.cpu().numpy()

        def lines(env, points, color, width=0.10):
            points = np.asarray(points, dtype=np.float32)
            if len(points) < 2 or not np.isfinite(points).all():
                return
            start, finish = points[:-1], points[1:]
            delta = finish - start
            # Isaac Gym add_lines has no line-width argument.  Overlapping
            # parallel segments create a visible world-space ribbon instead
            # of a one-pixel thread, including at a distant top camera.
            normal = np.column_stack((-delta[:, 1], delta[:, 0],
                                      np.zeros(len(delta))))
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / np.maximum(norm, 1e-6)
            vertical = norm[:, 0] < 1e-6
            normal[vertical] = (1.0, 0.0, 0.0)
            offsets = np.linspace(-width / 2, width / 2,
                                  max(3, int(np.ceil(width / 0.008)) + 1))
            shift = offsets[:, None, None] * normal[None]
            segments = np.concatenate((start[None] + shift, finish[None] + shift),
                                      axis=-1).reshape(-1, 6).astype(np.float32)
            colors = np.tile(np.asarray(color, dtype=np.float32), (len(segments), 1))
            self.gym.add_lines(self.viewer, env, len(segments), segments, colors)

        def lifted(points, z):
            return np.column_stack((points, np.full(len(points), z)))

        for e, env in enumerate(self.envs):
            for a in range(self.num_agents):
                lines(env, lifted(raw[e, a], 0.14 + 0.02 * a), (1.0, 1.0, 1.0),
                      width=0.10)
            if not self._carry_rehearsal[e].item():
                r = e * self.num_agents
                # Match observation-space position, size and yaw footprint,
                # without moving or creating any physical simulator actor.
                boxes = self.humanoid_rows(self._box_states)
                pos = virtual[e].copy()
                yaw = self._yaw(boxes[r:r + 1])[0].item()
                c, s = np.cos(yaw), np.sin(yaw)
                corners = np.array([[x, y, z] for z in (-1, 1)
                                    for y in (-1, 1) for x in (-1, 1)], dtype=np.float32)
                corners *= sizes[r] / 2
                xy = corners[:, :2].copy()
                corners[:, 0] = c * xy[:, 0] - s * xy[:, 1]
                corners[:, 1] = s * xy[:, 0] + c * xy[:, 1]
                corners += pos
                for i in range(8):
                    for bit in (1, 2, 4):
                        j = i ^ bit
                        if i < j:
                            lines(env, corners[[i, j]], (0.1, 0.5, 1.0), width=0.05)

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
            reset_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            reset_mask[env_ids] = True
            self._stack_history.reset(reset_mask)
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
