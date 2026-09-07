"""Learned joint planner in front of the frozen ms18 steer executor.

Only the inherited ``_gt_path/_mscale/_s_end`` provider changes. The policy
observation layout, reward, network, and executor checkpoint remain exactly the
HumanoidMASteerCarry contract.
"""

from __future__ import annotations

import os
import atexit
from pathlib import Path

import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import HumanoidMASteerCarry
from tokenhsi.utils import steer_path as sp
from trajectory_predictor.checkpoint import load_checkpoint
from trajectory_predictor.geometry import resample_coarse
from trajectory_predictor.oracle import solve_oracle
from trajectory_predictor.schema import AGENTS, SPEED_VALUES, PlannerState


class HumanoidMAPlannerCarry(HumanoidMASteerCarry):
    """A=2 wrapper whose joint plan is refreshed from actual simulator state."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._planner_ready = False
        self._planner_provider = os.environ.get("MS_PLANNER_PROVIDER", "learned").lower()
        if self._planner_provider not in ("learned", "oracle", "analytic", "gt"):
            raise ValueError("MS_PLANNER_PROVIDER must be learned, oracle, analytic, or gt")
        self._planner_replan_steps = int(os.environ.get("MS_REPLAN_STEPS", "6"))
        if self._planner_replan_steps <= 0:
            raise ValueError("MS_REPLAN_STEPS must be positive")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self.num_agents != AGENTS:
            raise ValueError(f"HumanoidMAPlannerCarry V1 requires exactly {AGENTS} agents")

        self._planner = None
        self._planner_checkpoint = None
        if self._planner_provider == "learned":
            raw_path = os.environ.get("MS_PLANNER_CKPT")
            if not raw_path:
                raise ValueError("MS_PLANNER_CKPT is required for MS_PLANNER_PROVIDER=learned")
            checkpoint = self._resolve_planner_checkpoint(raw_path)
            self._planner, self._planner_checkpoint = load_checkpoint(checkpoint, self.device)
            self._planner.eval()
            print(f"[planner] loaded {checkpoint}  replan={self._planner_replan_steps} action steps", flush=True)
        else:
            print(f"[planner] provider={self._planner_provider}  replan={self._planner_replan_steps}", flush=True)

        self._planner_phase = torch.zeros(self.num_envs, AGENTS, device=self.device)
        self._planner_last_replan = torch.full(
            (self.num_envs,), -self._planner_replan_steps, device=self.device, dtype=torch.long
        )
        self._planner_has_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._planner_invalid_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._planner_fallback_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._planner_replan_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._planner_episode_serial = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._planner_dagger_out = os.environ.get("MS_PLANNER_DAGGER_OUT", "")
        self._planner_dagger_max = int(os.environ.get("MS_PLANNER_DAGGER_MAX", "1000000"))
        self._planner_dagger_count = 0
        self._planner_dagger = []
        if self._planner_dagger_out:
            atexit.register(self._save_dagger_states)
        if self._planner_provider != "gt":
            atexit.register(self._report_planner_runtime)
        self._planner_ready = True
        if self._planner_provider != "gt":
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._planner_phase[:] = self._measured_phase(env_ids)
            self._plan_envs(env_ids)

    @staticmethod
    def _resolve_planner_checkpoint(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend((Path.cwd().parent / path, Path.cwd() / path))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"planner checkpoint not found: {raw_path}")

    def _measured_phase(self, env_ids):
        rows = self.agent_rows(env_ids)
        box = self.humanoid_rows(self._box_states)[rows].reshape(len(env_ids), AGENTS, -1)
        size_z = self._box_lib._box_size[rows, 2].reshape(len(env_ids), AGENTS)
        return (box[..., 2] > size_z / 2.0 + 0.2).float()

    def _planner_state(self, env_ids) -> PlannerState:
        rows = self.agent_rows(env_ids)
        n = len(env_ids)
        humanoid = self.humanoid_rows(self._humanoid_root_states)[rows].reshape(n, AGENTS, -1)
        box = self.humanoid_rows(self._box_states)[rows].reshape(n, AGENTS, -1)
        goal = self._box_tar_pos[rows, :2].reshape(n, AGENTS, 2)
        heading = torch.atan2(
            2.0 * (humanoid[..., 6] * humanoid[..., 5] + humanoid[..., 3] * humanoid[..., 4]),
            1.0 - 2.0 * (humanoid[..., 4] ** 2 + humanoid[..., 5] ** 2),
        )
        return PlannerState(
            root_xy=humanoid[..., :2],
            heading=heading,
            root_vel_xy=humanoid[..., 7:9],
            box_xyz=box[..., :3],
            box_vel_xy=box[..., 7:9],
            goal_xy=goal,
            phase=self._planner_phase[env_ids],
        )

    @staticmethod
    def _analytic_coarse(state: PlannerState) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, 17, device=state.device, dtype=state.root_xy.dtype)
        first = state.root_xy[:, :, None] + t[None, None, :, None] * (
            state.box_xyz[..., :2] - state.root_xy
        )[:, :, None]
        second = state.box_xyz[..., :2][:, :, None] + t[None, None, :, None] * (
            state.goal_xy - state.box_xyz[..., :2]
        )[:, :, None]
        return torch.cat((first, second[:, :, 1:]), dim=2)

    def _validate_plan(self, state, coarse, speed_class, extra_valid=None):
        finite = torch.isfinite(coarse).all(dim=(1, 2, 3))
        if extra_valid is not None:
            finite &= extra_valid
        safe_coarse = torch.nan_to_num(coarse, nan=0.0, posinf=0.0, neginf=0.0)
        anchors = (coarse[:, :, 0] - state.root_xy).norm(dim=-1).amax(dim=1) <= 0.01
        anchors &= (coarse[:, :, 16] - state.box_xyz[..., :2]).norm(dim=-1).amax(dim=1) <= 0.01
        anchors &= (coarse[:, :, 32] - state.goal_xy).norm(dim=-1).amax(dim=1) <= 0.01
        seg_len = (safe_coarse[:, :, 1:] - safe_coarse[:, :, :-1]).norm(dim=-1).sum(dim=-1)
        buffer_ok = (seg_len < (sp.V - 1) * sp.DS).all(dim=1)
        dense, end_s = resample_coarse(safe_coarse, with_end=True)
        flat = dense.reshape(-1, sp.V, 2)
        box = state.box_xyz[..., :2].reshape(-1, 2)
        s_box, _, _ = sp.project(box, flat)
        turn = sp.path_stats(flat, s_box)["turn_1.5m_deg"].reshape(-1, AGENTS)
        curvature_ok = (turn <= 35.0 + 1e-3).all(dim=1)
        class_ok = ((speed_class >= 0) & (speed_class < len(SPEED_VALUES))).all(dim=(1, 2))
        return finite & anchors & buffer_ok & curvature_ok & class_ok, dense, end_s, s_box.reshape(-1, AGENTS)

    def _install_plan(self, env_ids, dense, end_s, s_box, speed_class):
        rows = self.agent_rows(env_ids)
        flat_dense = dense.reshape(-1, sp.V, 2)
        flat_end = end_s.reshape(-1)
        flat_box = s_box.reshape(-1)
        flat_speed = speed_class.reshape(-1, 4)
        self._gt_path[rows] = flat_dense
        self._s_end[rows] = flat_end
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = flat_box
        self._prev_arc[rows] = 0.0

        cell_s = torch.arange(sp.V, device=self.device, dtype=flat_end.dtype)[None] * sp.DS
        quarter = torch.floor(4.0 * cell_s / flat_end[:, None].clamp(min=1e-6)).long().clamp(0, 3)
        selected = flat_speed.gather(1, quarter)
        speed_values = torch.tensor(SPEED_VALUES, device=self.device, dtype=flat_end.dtype)
        # inherited v_cmd = 1.5 * mscale
        self._mscale[rows] = speed_values[selected] / 1.5

    @torch.no_grad()
    def _plan_envs(self, env_ids):
        if len(env_ids) == 0 or self._planner_provider == "gt":
            return
        state = self._planner_state(env_ids)
        self._collect_dagger_states(state, env_ids)
        if self._planner_provider == "analytic":
            coarse = self._analytic_coarse(state)
            speed_class = torch.full((len(env_ids), AGENTS, 4), 3, device=self.device, dtype=torch.long)
            logits_finite = None
        elif self._planner_provider == "oracle":
            oracle = solve_oracle(state.to("cpu"))
            coarse = oracle.coarse_path.to(self.device)
            speed_class = oracle.speed_class.to(self.device)
            logits_finite = oracle.oracle_valid.to(self.device)
        else:
            with torch.inference_mode():
                output = self._planner(state)
            coarse = output["coarse_world"]
            speed_class = output["speed_logits"].argmax(dim=-1)
            logits_finite = torch.isfinite(output["speed_logits"]).all(dim=(1, 2, 3))
        valid, dense, end_s, s_box = self._validate_plan(
            state, coarse, speed_class, extra_valid=logits_finite
        )
        self._planner_replan_count[env_ids] += 1
        self._planner_invalid_count[env_ids] += (~valid).long()

        accept = valid
        if accept.any():
            chosen = env_ids[accept]
            self._install_plan(chosen, dense[accept], end_s[accept], s_box[accept], speed_class[accept])
            self._planner_has_valid[chosen] = True

        need_fallback = (~accept) & (~self._planner_has_valid[env_ids])
        if need_fallback.any():
            fallback_state = state.index(need_fallback)
            fallback = self._analytic_coarse(fallback_state)
            fallback_speed = torch.full(
                (int(need_fallback.sum()), AGENTS, 4), 3, device=self.device, dtype=torch.long
            )
            _, fb_dense, fb_end, fb_box = self._validate_plan(fallback_state, fallback, fallback_speed)
            chosen = env_ids[need_fallback]
            self._install_plan(chosen, fb_dense, fb_end, fb_box, fallback_speed)
            self._planner_has_valid[chosen] = True
            self._planner_fallback_count[chosen] += 1
        # Invalid replans after a valid one intentionally leave the previous
        # joint path and speed profile untouched.
        self._planner_last_replan[env_ids] = self.progress_buf[env_ids]

    def _collect_dagger_states(self, state, env_ids):
        if not self._planner_dagger_out or self._planner_provider != "learned":
            return
        remaining = self._planner_dagger_max - self._planner_dagger_count
        if remaining <= 0:
            return
        keep = min(state.batch_size, remaining)
        state = state.index(slice(0, keep)).to("cpu")
        chosen = env_ids[:keep]
        episode_key = chosen.long() + self.num_envs * self._planner_episode_serial[chosen]
        scenario = torch.full(
            (keep,), 1 if self.scen == "cross" else 0, dtype=torch.uint8
        )
        self._planner_dagger.append({
            "state": state.as_dict(),
            "episode_key": episode_key.cpu(),
            "step": self.progress_buf[chosen].cpu().clone(),
            "scenario": scenario,
        })
        self._planner_dagger_count += keep

    def _save_dagger_states(self):
        if not self._planner_dagger_out or not self._planner_dagger:
            return
        output = Path(self._planner_dagger_out).expanduser()
        if not output.is_absolute():
            output = Path.cwd().parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        state = {
            key: torch.cat([chunk["state"][key] for chunk in self._planner_dagger], dim=0)
            for key in self._planner_dagger[0]["state"]
        }
        payload = {
            "schema_version": "masteer-dagger-state-v1",
            "state": state,
            "episode_key": torch.cat([chunk["episode_key"] for chunk in self._planner_dagger]),
            "step": torch.cat([chunk["step"] for chunk in self._planner_dagger]),
            "scenario": torch.cat([chunk["scenario"] for chunk in self._planner_dagger]),
        }
        torch.save(payload, output)
        print(f"[planner] saved {self._planner_dagger_count} DAgger states to {output}", flush=True)

    def _report_planner_runtime(self):
        replans = int(self._planner_replan_count.sum().item())
        invalid = int(self._planner_invalid_count.sum().item())
        fallback = int(self._planner_fallback_count.sum().item())
        rate = invalid / max(replans, 1)
        print(
            f"MS_PLANNER_SUMMARY provider={self._planner_provider} replans={replans} "
            f"invalid={invalid} invalid_rate={rate:.6f} analytic_fallback={fallback}",
            flush=True,
        )

    def _reset_steer(self, rows):
        # Preserve every inherited reset-side metric and GT-provider behavior.
        super()._reset_steer(rows)
        if not getattr(self, "_planner_ready", False) or self._planner_provider == "gt" or len(rows) == 0:
            return
        env_ids = torch.div(rows, AGENTS, rounding_mode="floor").unique()
        self._planner_episode_serial[env_ids] += 1
        self._planner_has_valid[env_ids] = False
        self._planner_phase[env_ids] = self._measured_phase(env_ids)
        self._planner_last_replan[env_ids] = -self._planner_replan_steps
        self._plan_envs(env_ids)

    def _maybe_replan(self, env_ids):
        if not self._planner_ready or self._planner_provider == "gt" or len(env_ids) == 0:
            return
        env_ids = env_ids.unique()
        measured = self._measured_phase(env_ids)
        old_phase = self._planner_phase[env_ids]
        new_phase = torch.maximum(old_phase, measured)
        phase_changed = (new_phase != old_phase).any(dim=1)
        self._planner_phase[env_ids] = new_phase
        due = (self.progress_buf[env_ids] - self._planner_last_replan[env_ids]) >= self._planner_replan_steps
        selected = env_ids[due | phase_changed]
        self._plan_envs(selected)

    def _compute_task_obs(self, env_ids=None):
        active = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._maybe_replan(active)
        return super()._compute_task_obs(env_ids)
