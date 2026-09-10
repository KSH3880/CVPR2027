"""Opt-in learned path/speed planner over the existing sequential-stack task."""

import os
from pathlib import Path

import torch

from coordinator.planner import apply_fixed_priority
from coordinator.schema import CoordinatorState, MAX_SPEED, MIN_SPEED, PATH_DS, PATH_VERTICES
from coordinator.sequential_bridge import carry_rows, load_planner, resample_plan, select_stack_plan
from env.tasks.adapt_interaction_skills.humanoid_ma_sequential_stack_carry import (
    HumanoidMASequentialStackCarry,
)
from tokenhsi.utils import steer_path as sp


class HumanoidMACoordSequentialStack(HumanoidMASequentialStackCarry):
    """Attach state-only planning to stack phases without changing policy ABI."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._coord_ready = False
        raw = os.environ.get("COORD_CKPT", "")
        if not raw or not Path(raw).expanduser().is_file():
            raise FileNotFoundError("COORD_CKPT must name an existing coordinator PTH")
        self._coord_period = int(os.environ.get("COORD_REPLAN_STEPS", "6"))
        self._coord_goal_tol = float(os.environ.get("COORD_GOAL_REPLAN_M", "0.05"))
        self._coord_accel = float(os.environ.get("COORD_COMMAND_ACCEL", "0.75"))
        if self._coord_period < 1 or self._coord_goal_tol <= 0 or self._coord_accel <= 0:
            raise ValueError("coordinator replan interval, goal tolerance and acceleration must be positive")
        if cfg["env"].get("numAgents", 1) != 2:
            raise ValueError("coordinator sequential stack requires numAgents=2")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if sp.DS != PATH_DS or sp.V != PATH_VERTICES or self.steer_dim() != 12:
            raise ValueError("coordinator needs ms18's 0.1m/320-point/12-D steer contract")
        if not self.steer_clip or self.steer_zero:
            raise ValueError("coordinator execution requires MS_CLIP=1 and MS_ZERO=0")
        self._coordinator, payload = load_planner(raw, self.device)
        self._coord_random_priority = bool(payload.get("extras", {}).get("random_priority", False))
        self._coord_learned_priority = bool(getattr(self._coordinator.config, "learned_priority", False))
        if self._coord_random_priority and self._coord_learned_priority:
            raise ValueError("checkpoint enables both random and learned priority")
        self._coord_priority = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._coord_priority_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._coord_tick = torch.full((self.num_envs,), -self._coord_period, device=self.device, dtype=torch.long)
        self._coord_phase = torch.full((self.num_envs,), -99, device=self.device, dtype=torch.long)
        self._coord_carry_phase = torch.full((self.num_envs, 2), -1., device=self.device)
        self._coord_goal = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._coord_active = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._coord_applied = torch.zeros(self._rows, dtype=torch.bool, device=self.device)
        self._coord_speed = torch.full((self._rows,), MAX_SPEED, device=self.device)
        self._coord_speed_tick = torch.full((self._rows,), -1, dtype=torch.long, device=self.device)
        self._coord_replans = self._coord_invalid = self._coord_fallback = 0
        self._coord_installed = self._coord_unsafe = 0
        self._coord_ready = True
        self._coord_reset(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        print("[coord-stack] checkpoint={} schema={} step={} replan={} random_priority={} "
              "learned_priority={} frozen=True".format(
                  Path(raw).expanduser().resolve(), payload["schema_version"], payload.get("step"),
                  self._coord_period, self._coord_random_priority, self._coord_learned_priority), flush=True)

    @staticmethod
    def _coord_yaw(states):
        return torch.atan2(2 * (states[..., 6] * states[..., 5] + states[..., 3] * states[..., 4]),
                           1 - 2 * (states[..., 4].square() + states[..., 5].square()))

    def coord_state(self, env_ids):
        rows = self.agent_rows(env_ids)
        root = self.humanoid_rows(self._humanoid_root_states)[rows].reshape(-1, 2, 13)
        # Clone because retreat planning rewrites A1's box only in the planner
        # snapshot; this tensor must never alias Isaac Gym's physical state.
        box = self.humanoid_rows(self._box_states)[rows].reshape(-1, 2, 13).clone()
        size = self._box_lib._box_size[rows].reshape(-1, 2, 3)
        goal = self._box_tar_pos[rows, :2].reshape(-1, 2, 2).clone()
        # Same measured lift/reference-floor criterion as the stack executor.
        bottom = box[..., 2] - size[..., 2] / 2
        lifted = bottom > self._initial_box_bottom_z[rows].reshape(-1, 2) + 0.08
        near = (root[..., :2] - box[..., :2]).norm(dim=-1) <= 0.7
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands_on = self._handheld_score(root.reshape(-1, 13)[:, :3],
                                        box.reshape(-1, 13)[:, :3], rigid).reshape(-1, 2) > 0.35
        held = lifted & hands_on
        at_goal = (box[..., :2] - goal).norm(dim=-1) <= 0.15
        phase = torch.where(held, torch.full_like(bottom, 2.), near.float())
        phase = torch.where(at_goal & ~held, torch.full_like(phase, 3.), phase)

        # During A1 retreat the frozen Carry executor sees a virtual stationary
        # box at the retreat endpoint. Give the planner the exact same world
        # state, so its root->box leg becomes the retreat path and its
        # box->goal leg is degenerate. The placed physical box remains untouched.
        retreat = (self._stack_phase[env_ids] == self.A1_RETREAT) & ~self._carry_rehearsal[env_ids]
        if retreat.any():
            virtual_xy = self._a1_retreat_pos[env_ids[retreat], :2]
            box[retreat, 0, :2] = virtual_xy
            box[retreat, 0, 7:9] = 0
            goal[retreat, 0] = virtual_xy
            held[retreat, 0] = False
            near_virtual = ((root[retreat, 0, :2] - virtual_xy).norm(dim=-1) <= 0.7)
            phase[retreat, 0] = near_virtual.float()
        return CoordinatorState(root[..., :2], self._coord_yaw(root), root[..., 7:9],
                                box[..., :3], self._coord_yaw(box), box[..., 7:9],
                                size[..., :2], goal, held.float(), phase)

    def _coord_reset(self, env_ids):
        rows = self.agent_rows(env_ids)
        self._coord_tick[env_ids] = -self._coord_period
        self._coord_phase[env_ids] = -99
        self._coord_applied[rows] = False
        self._coord_active[env_ids] = False
        self._coord_priority_locked[env_ids] = False
        if self._coord_random_priority:
            self._coord_priority[env_ids] = torch.randint(2, (len(env_ids),), device=self.device)
        root = self.humanoid_rows(self._humanoid_root_states)[rows]
        self._coord_speed[rows] = root[:, 7:9].norm(dim=-1).clamp(MIN_SPEED, MAX_SPEED)
        self._coord_speed_tick[rows] = -1

    def _reset_envs(self, env_ids):
        ready = getattr(self, "_coord_ready", False)
        self._coord_ready = False
        try:
            super()._reset_envs(env_ids)
        finally:
            self._coord_ready = ready
        if ready and len(env_ids):
            self._coord_reset(env_ids)
            # The parent constructed its reset observation before our state was reset.
            self._compute_observations(env_ids)

    def _coord_install(self, env_ids, path, speed, active, planner_box_xy):
        dense, dense_speed, end = resample_plan(path, speed)
        mask = active.reshape(-1)
        rows = self.agent_rows(env_ids)[mask]
        if not len(rows):
            return
        flat = dense.reshape(-1, sp.V, 2)[mask]
        self._gt_path[rows] = flat
        self._mscale[rows] = dense_speed.reshape(-1, sp.V)[mask] / MAX_SPEED
        self._s_end[rows] = end.reshape(-1)[mask]
        self._arc_root[rows] = 0
        self._prev_arc[rows] = 0
        boxes = planner_box_xy.reshape(-1, 2)[mask]
        self._arc_box[rows] = sp.project(boxes, flat)[0]
        self._coord_applied[rows] = True
        self._coord_installed += len(rows)

    @torch.no_grad()
    def _coord_plan(self, env_ids):
        if not len(env_ids):
            return
        state = self.coord_state(env_ids)
        phase = self._stack_phase[env_ids]
        active = carry_rows(phase, self._carry_rehearsal[env_ids])
        changed = ((phase != self._coord_phase[env_ids])
                   | (active != self._coord_active[env_ids]).any(dim=-1)
                   | ((state.goal_xy - self._coord_goal[env_ids]).norm(dim=-1) > self._coord_goal_tol).any(dim=-1))
        due = self.progress_buf[env_ids] - self._coord_tick[env_ids] >= self._coord_period
        due |= changed | (state.phase != self._coord_carry_phase[env_ids]).any(dim=-1)
        if not due.any():
            return
        env_ids, state, active, phase, changed = env_ids[due], state.index(due), active[due], phase[due], changed[due]
        # Phase ownership is independent of checkpoint priority and never changes task order.
        changed_rows = self.agent_rows(env_ids[changed])
        self._coord_applied[changed_rows] = False
        root = self.humanoid_rows(self._humanoid_root_states)[changed_rows]
        self._coord_speed[changed_rows] = root[:, 7:9].norm(dim=-1).clamp(MIN_SPEED, MAX_SPEED)
        self._coord_speed_tick[changed_rows] = -1
        inactive = self.agent_rows(env_ids)[~active.reshape(-1)]
        self._coord_applied[inactive] = False
        moving = active.any(dim=-1)
        if moving.any():
            ids, current, use = env_ids[moving], state.index(moving), active[moving]
            output = self._coordinator(current)
            if self._coord_learned_priority:
                unlocked = ~self._coord_priority_locked[ids]
                self._coord_priority[ids[unlocked]] = output["priority_logits"].argmax(dim=-1)[unlocked]
                self._coord_priority_locked[ids] = True
            if self._coord_random_priority or self._coord_learned_priority:
                output = apply_fixed_priority(output, current, self._coord_priority[ids])
            path, speed, valid, safe, _ = select_stack_plan(output, current, use)
            self._coord_replans += len(ids)
            self._coord_invalid += int((~valid).sum())
            self._coord_unsafe += int((valid & ~safe).sum())
            if valid.any():
                self._coord_install(ids[valid], path[valid], speed[valid], use[valid],
                                    current.box_xyz[valid, :, :2])
            # First/changed-goal invalid plans use a fresh native plan. Same-context
            # invalid replans retain the previous accepted plan.
            bad_rows = self.agent_rows(ids[~valid])[use[~valid].reshape(-1)]
            fallback = bad_rows[~self._coord_applied[bad_rows]]
            if len(fallback):
                # Fail closed to the same skill goal the planner saw. This is
                # the virtual retreat box for A1/phase2 and the physical stack
                # goal for ordinary Carry phases.
                flat_goal = current.goal_xy[~valid].reshape(-1, 2)[use[~valid].reshape(-1)]
                missing = ~self._coord_applied[bad_rows]
                target = torch.cat((flat_goal[missing],
                                    self._box_tar_pos[fallback, 2:3]), dim=-1)
                self._reset_steer_to(fallback, target)
                self._coord_fallback += len(fallback)
        self._coord_tick[env_ids] = self.progress_buf[env_ids]
        self._coord_phase[env_ids] = phase
        self._coord_carry_phase[env_ids] = state.phase
        self._coord_goal[env_ids] = state.goal_xy
        self._coord_active[env_ids] = active

    def _coord_update_speed(self, rows):
        tick = self.progress_rows()[rows]
        selected = self._coord_applied[rows] & (tick != self._coord_speed_tick[rows])
        rows, tick = rows[selected], tick[selected]
        if not len(rows):
            return
        # Read the planned profile directly so this cannot recurse through _m_at.
        j = (self._arc_root[rows] / sp.DS).long().clamp(0, sp.V - 1)
        desired = MAX_SPEED * self._mscale[rows, j]
        delta = (desired - self._coord_speed[rows]).clamp(-self._coord_accel * self.dt,
                                                       self._coord_accel * self.dt)
        self._coord_speed[rows] = (self._coord_speed[rows] + delta).clamp(MIN_SPEED, MAX_SPEED)
        self._coord_speed_tick[rows] = tick

    def _m_at(self, arc, rows=None):
        nominal = super()._m_at(arc, rows)
        if not getattr(self, "_coord_ready", False):
            return nominal
        rows = torch.arange(len(arc), device=self.device) if rows is None else rows
        return torch.where(self._coord_applied[rows], 1.6 * self._coord_speed[rows], nominal)

    def _compute_task_obs(self, env_ids=None):
        if getattr(self, "_coord_ready", False):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            self._coord_plan(ids)
            self._coord_update_speed(self.agent_rows(ids))
        return super()._compute_task_obs(env_ids)

    def report_metrics(self):
        super().report_metrics()
        if getattr(self, "_coord_ready", False):
            print("COORD_STACK_SUMMARY replans={} invalid={} unsafe={} installed_agent_rows={} "
                  "fallback_agent_rows={}".format(
                      self._coord_replans, self._coord_invalid, self._coord_unsafe,
                      self._coord_installed, self._coord_fallback), flush=True)
