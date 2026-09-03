"""Strict three-step stacking using the existing 340-D MA-steer policy ABI.

Sequence:
  1. A1 places the bottom box.
  2. A1 follows a steering command away from the placed box.
  3. Only after A1 is clear, A2 receives the carry goal directly above the
     measured bottom-box pose and starts its carry/placement task.

Unlike the original stack coordinator, A2 does not travel to a staging point
while A1 is working and the future top goal is not exposed during A1 retreat.
"""

import os

import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_stack_carry import (
    HumanoidMAStackCarry,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_carry import (
    CARRY_HI,
    CARRY_LO,
    TEAMMATE_DIM,
)
from isaacgym.torch_utils import quat_mul, quat_rotate
from isaacgym import gymtorch
from utils import torch_utils


class HumanoidMASequentialStackCarry(HumanoidMAStackCarry):
    """A1 putdown -> A1 retreat -> A2 carry-to-top, with a 340-D ABI."""

    PHASE_NAMES = (
        "A1_PLACE", "VERIFY_BOTTOM", "A1_RETREAT",
        "A2_RESUME", "VERIFY_STACK", "DONE",
    )

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        # The inherited 600-step horizon is sized for one carry episode.  This
        # task executes two carries plus a retreat sequentially, so that limit
        # resets otherwise healthy episodes halfway through A2's turn.
        horizon = int(os.environ.get("STACK_EPISODE_LENGTH", "900"))
        cfg["env"]["episodeLength"] = horizon
        cfg["env"]["episodeLengthShort"] = horizon
        # Sequential handoff does not require pixel-perfect bottom placement.
        # Once the box is near the floor goal and physically stable, open the
        # retreat phase sooner than the original stack training environment.
        os.environ.setdefault("STACK_POS_TOL", "0.30")
        super().__init__(cfg=cfg, sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id,
                         headless=headless)
        self._sequential_reset_reported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

    def _set_phase(self, env_ids, phase):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if (int(os.environ.get("STACK_DEBUG", "0")) != 0
                and hasattr(self, "_stack_phase")):
            changed = env_ids[self._stack_phase[env_ids] != phase]
            if bool((changed == 0).any()):
                old = int(self._stack_phase[0])
                print("[sequential-stack] env0 {} -> {} age={}".format(
                    self.PHASE_NAMES[old], self.PHASE_NAMES[phase],
                    int(self._stack_phase_age[0])), flush=True)
        return super()._set_phase(env_ids, phase)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if hasattr(self, "_sequential_reset_reported") and len(env_ids) > 0:
            self._sequential_reset_reported[env_ids] = False

    def _compute_reset(self):
        super()._compute_reset()
        if (int(os.environ.get("STACK_DEBUG", "0")) != 0
                and hasattr(self, "_sequential_reset_reported")):
            ended = self.reset_buf.bool() & ~self._sequential_reset_reported
            if bool((ended & (torch.arange(
                    self.num_envs, device=self.device) == 0)).any()):
                timeout = int(self.progress_buf[0]) >= self.max_episode_length - 1
                fallen = bool(self._terminate_buf[0])
                print("[sequential-stack] env0 reset: phase={} step={} "
                      "timeout={} fall={}".format(
                          self.PHASE_NAMES[int(self._stack_phase[0])],
                          int(self.progress_buf[0]), timeout, fallen),
                      flush=True)
            self._sequential_reset_reported |= ended

    def post_physics_step(self):
        super().post_physics_step()
        # Full-stack mode in the parent leaves DONE alive until the global
        # timeout.  End this sequential episode immediately and normally once
        # the top box has remained stable for the requested number of steps.
        self.reset_buf[self._stack_phase == self.DONE] = 1

    def _virtual_retreat_carry_obs(self, rows, env_ids):
        """Give A1 a virtual carried box while leaving physics untouched."""
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        real_box = self.humanoid_rows(self._box_states)[rows]
        heading_inv = torch_utils.calc_heading_quat_inv(roots[:, 3:7])

        # Pretend the released box is still centered between A1's hands.
        box_pos = rigid[:, self._key_body_ids[[0, 1]]].mean(dim=1)
        box_rot = real_box[:, 3:7]
        box_vel = roots[:, 7:10]
        box_ang_vel = torch.zeros_like(box_vel)

        local_box_vel = quat_rotate(heading_inv, box_vel)
        local_box_ang_vel = quat_rotate(heading_inv, box_ang_vel)
        local_box_pos = quat_rotate(heading_inv, box_pos - roots[:, 0:3])
        local_box_rot = torch_utils.quat_to_tan_norm(
            quat_mul(heading_inv, box_rot))

        bps = self._box_lib._box_bps[rows]
        n = bps.shape[1]
        box_rot_exp = box_rot[:, None, :].expand(-1, n, -1)
        world_bps = (quat_rotate(
            box_rot_exp.reshape(-1, 4), bps.reshape(-1, 3))
            .view(len(rows), n, 3) + box_pos[:, None, :])
        heading_exp = heading_inv[:, None, :].expand(-1, n, -1)
        root_exp = roots[:, None, 0:3].expand(-1, n, -1)
        local_bps = quat_rotate(
            heading_exp.reshape(-1, 4),
            (world_bps - root_exp).reshape(-1, 3)).view(len(rows), -1)

        # Carry the virtual box toward the retreat endpoint, at hand height,
        # so the old carry token requests transport rather than another drop.
        carry_goal = self._a1_retreat_pos[env_ids].clone()
        carry_goal[:, 2] = box_pos[:, 2]
        local_goal = quat_rotate(
            heading_inv, carry_goal - roots[:, 0:3])

        obs = torch.cat((local_box_vel, local_box_ang_vel, local_box_pos,
                         local_box_rot, local_bps, local_goal), dim=-1)
        assert obs.shape[-1] == CARRY_HI - CARRY_LO, obs.shape
        return obs

    def _compute_task_obs(self, env_ids=None):
        obs = super()._compute_task_obs(env_ids)
        if (not hasattr(self, "_stack_phase")
                or int(os.environ.get("STACK_VIRTUAL_RETREAT_BOX", "1")) == 0):
            return obs

        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        agent = rows % self.num_agents
        # Once A1 has released the bottom box, never expose that real box to
        # its carry token again.  Restricting this to A1_RETREAT made A1 see
        # the placed box as soon as the phase advanced to A2_RESUME, so the
        # pretrained carry policy naturally walked back to pick it up.
        parked_a1 = ((agent == 0)
                     & (self._stack_phase[env] >= self.A1_RETREAT))
        if not parked_a1.any():
            return obs

        virtual = self._virtual_retreat_carry_obs(
            rows[parked_a1], env[parked_a1])
        out = obs.clone()
        carry_start = TEAMMATE_DIM * (self.num_agents - 1) + self.steer_dim()
        carry_dim = CARRY_HI - CARRY_LO
        out[parked_a1, carry_start:carry_start + carry_dim] = virtual
        out[parked_a1,
            carry_start + carry_dim:carry_start + 2 * carry_dim] = virtual
        return out

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if not hasattr(self, "_stack_phase") or len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rows = self.agent_rows(env_ids).view(-1, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        boxes = self.humanoid_rows(self._box_states)

        # The parent carry reset may sample an elevated target and place a
        # physical target platform under it.  Bottom-box placement in this
        # sequential task is always on the ground: its center z is exactly
        # ground_z + half box height.
        ground_z = float(os.environ.get("STACK_GROUND_Z", "0.0"))
        bottom = self._box_tar_pos[r0].clone()
        bottom[:, 2] = (ground_z
                        + 0.5 * self._box_lib._box_size[r0, 2])
        self._bottom_nominal_pos[env_ids] = bottom
        self._committed_bottom_pos[env_ids] = bottom
        self._box_tar_pos[r0] = bottom
        self._reset_steer_to(r0, bottom)

        # Remove the random-height target platform from physics as well as
        # from the command.  The real ground plane is the only support.
        if self._carry_reset_random_height:
            target_platform = self.agent_axis(self._tar_platform_pos)
            target_platform[env_ids, 0, 2] = -10.0
            actor_ids = self._tar_platform_actor_ids[r0]
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

        # A2 waits where its own box was reset.  In particular it does not
        # receive the original coordinator's pre-handoff staging command.
        wait_goal = boxes[r1, 0:3].clone()
        self._a2_stage_pos[env_ids] = wait_goal
        self._committed_top_pos[env_ids] = wait_goal
        self._top_committed[env_ids] = False
        self._box_tar_pos[r1] = wait_goal
        self._reset_steer_to(r1, wait_goal)
        self._prev_stage_dist[env_ids] = 0.0
        self._prev_top_dist[env_ids] = 0.0

    def _commit_top_goal(self, env_ids):
        """Compute the physical top pose without exposing it to A2 yet."""
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids).view(-1, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        boxes = self.humanoid_rows(self._box_states)
        size = self._box_lib._box_size
        bottom = boxes[r0]

        top = bottom[:, 0:3].clone()
        top[:, 2] += (0.5 * (size[r0, 2] + size[r1, 2])
                      + self.stack_top_clearance)
        self._committed_top_pos[env_ids] = top
        self._committed_bottom_pos[env_ids] = bottom[:, 0:3]
        q = bottom[:, 3:7]
        self._committed_top_yaw[env_ids] = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        self._top_committed[env_ids] = True

        # Deliberately do not modify _box_tar_pos[r1] here.  A2 continues to
        # see its wait goal throughout A1's retreat.

    def _activate_a2_top_goal(self, env_ids):
        """Open the dependency and start A2's ordinary carry task."""
        if len(env_ids) == 0:
            return
        r1 = self.agent_rows(env_ids).view(-1, 2)[:, 1]
        top = self._committed_top_pos[env_ids]
        self._box_tar_pos[r1] = top
        self._reset_steer_to(r1, top)
        boxes = self.humanoid_rows(self._box_states)
        roots = self.humanoid_rows(self._humanoid_root_states)
        self._prev_top_dist[env_ids] = torch.norm(
            boxes[r1, 0:3] - top, dim=-1)
        if int(os.environ.get("STACK_DEBUG", "0")) != 0:
            first = 0
            print("[sequential-stack] A2 activated: box_to_top={:.3f}m "
                  "root_to_box={:.3f}m top=({:+.2f},{:+.2f},{:+.2f})".format(
                      float(torch.norm(boxes[r1[first], 0:3] - top[first])),
                      float(torch.norm(roots[r1[first], 0:3]
                                       - boxes[r1[first], 0:3])),
                      float(top[first, 0]), float(top[first, 1]),
                      float(top[first, 2])), flush=True)

    def _update_stack_coordinator(self):
        """Advance only through the requested sequential dependency."""
        if not hasattr(self, "_stack_phase"):
            return
        self._stack_phase_age += 1
        rows = self.all_rows().view(self.num_envs, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        boxes = self.humanoid_rows(self._box_states)
        roots = self.humanoid_rows(self._humanoid_root_states)

        bottom_err = torch.norm(
            boxes[r0, 0:3] - self._bottom_nominal_pos, dim=-1)
        bottom_slow = (
            (torch.norm(boxes[r0, 7:10], dim=-1) < self.stack_vel_tol)
            & (torch.norm(boxes[r0, 10:13], dim=-1)
               < self.stack_ang_vel_tol))
        bottom_candidate = (bottom_err < self.stack_pos_tol) & bottom_slow
        self._bottom_stable_count = torch.where(
            bottom_candidate, self._bottom_stable_count + 1,
            torch.zeros_like(self._bottom_stable_count))

        phase = self._stack_phase
        verify = torch.nonzero(
            (phase == self.A1_PLACE) & (bottom_err < self.stack_pos_tol),
            as_tuple=False).squeeze(-1)
        self._set_phase(verify, self.VERIFY_BOTTOM)

        phase = self._stack_phase
        placed = torch.nonzero(
            (phase == self.VERIFY_BOTTOM)
            & (self._bottom_stable_count >= self.stack_stable_steps),
            as_tuple=False).squeeze(-1)
        if len(placed) > 0:
            # The top target is derived from the settled physical bottom box,
            # then kept private until A1 finishes retreating.
            self._commit_top_goal(placed)
            self._update_retreat_goal(placed)
            self._activate_retreat_steer(placed)
            self._set_phase(placed, self.A1_RETREAT)

        phase = self._stack_phase
        # Judge completion against the retreat command itself.  The old
        # root-to-box >= retreat_dist test could remain false even after A1
        # reached the generated path endpoint because its initial clearance,
        # release posture and the settled box center all vary by episode.
        retreat_rel = roots[r0, 0:2] - self._a1_retreat_start
        retreat_along = (retreat_rel * self._a1_retreat_dir).sum(dim=-1)
        retreat_goal_err = torch.norm(
            roots[r0, 0:2] - self._a1_retreat_pos[:, 0:2], dim=-1)
        release_tol = float(os.environ.get("STACK_RETREAT_RELEASE_TOL", "0.30"))
        min_progress = float(os.environ.get(
            "STACK_RETREAT_MIN_PROGRESS", "0.60"))
        clear = ((retreat_goal_err <= release_tol)
                 | (retreat_along >= min_progress))
        departed = torch.nonzero(
            (phase == self.A1_RETREAT) & clear,
            as_tuple=False).squeeze(-1)
        if len(departed) > 0:
            self._activate_a2_top_goal(departed)
            self._set_phase(departed, self.A2_RESUME)

        phase = self._stack_phase
        top_delta = boxes[r1, 0:3] - self._committed_top_pos
        top_xy_err = torch.norm(top_delta[:, 0:2], dim=-1)
        top_z_err = torch.abs(top_delta[:, 2])
        top_slow = (
            (torch.norm(boxes[r1, 7:10], dim=-1) < self.stack_vel_tol)
            & (torch.norm(boxes[r1, 10:13], dim=-1)
               < self.stack_ang_vel_tol))
        top_candidate = (
            (top_xy_err < self.stack_top_xy_tol)
            & (top_z_err < self.stack_top_z_tol) & top_slow)
        self._top_stable_count = torch.where(
            top_candidate, self._top_stable_count + 1,
            torch.zeros_like(self._top_stable_count))

        enter_verify = torch.nonzero(
            (phase == self.A2_RESUME)
            & (top_xy_err < self.stack_top_xy_tol)
            & (top_z_err < self.stack_top_z_tol),
            as_tuple=False).squeeze(-1)
        self._set_phase(enter_verify, self.VERIFY_STACK)

        phase = self._stack_phase
        lost = torch.nonzero(
            (phase == self.VERIFY_STACK) & ~top_candidate,
            as_tuple=False).squeeze(-1)
        self._set_phase(lost, self.A2_RESUME)
        done = torch.nonzero(
            (phase == self.VERIFY_STACK)
            & (self._top_stable_count >= self.stack_stable_steps),
            as_tuple=False).squeeze(-1)
        self._set_phase(done, self.DONE)
