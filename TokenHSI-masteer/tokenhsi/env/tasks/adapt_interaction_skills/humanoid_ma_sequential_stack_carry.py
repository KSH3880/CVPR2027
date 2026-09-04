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
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
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
        self.stack_carry_rehearsal_prob = float(os.environ.get(
            "STACK_CARRY_REHEARSAL_PROB", "0.0"))
        if not 0.0 <= self.stack_carry_rehearsal_prob <= 1.0:
            raise ValueError("STACK_CARRY_REHEARSAL_PROB must be in [0, 1]")
        # The first reset runs inside super().__init__.  None keeps that
        # construction reset on the old path; later resets sample per-env.
        self._carry_rehearsal = None
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
        self._carry_rehearsal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        # Episode-direct evaluation metrics.  Columns are grouped as
        # placement / retreat / A2 so a long wait in one phase cannot hide a
        # steering regression in another phase.  Shape is (agent rows, 3).
        self._seq_ep_lat_root = torch.zeros(
            (self._rows, 3), device=self.device)
        self._seq_ep_lat_box = torch.zeros(
            (self._rows, 3), device=self.device)
        self._seq_ep_steps = torch.zeros(
            (self._rows, 3), device=self.device, dtype=torch.long)
        self.stack_hand_clear_start = float(os.environ.get(
            "STACK_HAND_CLEAR_START", "0.08"))
        self.stack_hand_clear_done = float(os.environ.get(
            "STACK_HAND_CLEAR_DONE", "0.15"))
        self.stack_release_reward_w = float(os.environ.get(
            "STACK_RELEASE_REWARD_W", "0.50"))
        self.stack_early_release_penalty = float(os.environ.get(
            "STACK_EARLY_RELEASE_PENALTY", "0.50"))
        self.stack_hands_on_penalty = float(os.environ.get(
            "STACK_HANDS_ON_PENALTY", "0.35"))
        self.stack_foot_box_clearance = float(os.environ.get(
            "STACK_FOOT_BOX_CLEARANCE", "0.0"))
        self.stack_foot_box_penalty = float(os.environ.get(
            "STACK_FOOT_BOX_PENALTY", "0.0"))
        self.stack_bottom_z_tol = float(os.environ.get(
            "STACK_BOTTOM_Z_TOL", "0.05"))
        self.stack_release_grace_steps = int(float(os.environ.get(
            "STACK_RELEASE_GRACE_STEPS", "60")))
        self.stack_end_on_a2_resume = bool(int(os.environ.get(
            "STACK_END_ON_A2_RESUME", "0")))
        if self.stack_hand_clear_done <= self.stack_hand_clear_start:
            raise ValueError(
                "STACK_HAND_CLEAR_DONE must exceed STACK_HAND_CLEAR_START")
        if self.stack_bottom_z_tol <= 0.0:
            raise ValueError("STACK_BOTTOM_Z_TOL must be positive")
        if self.stack_release_grace_steps < self.stack_stable_steps:
            raise ValueError(
                "STACK_RELEASE_GRACE_STEPS must be >= STACK_STABLE_STEPS")

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

    def _reset_task_indicator(self, env_ids):
        """Sample full native-carry rehearsal episodes at every reset."""
        super()._reset_task_indicator(env_ids)
        if self._carry_rehearsal is None or len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._carry_rehearsal[env_ids] = (
            torch.rand(len(env_ids), device=self.device)
            < self.stack_carry_rehearsal_prob)

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
                phase_name = ("CARRY_REHEARSAL"
                              if self._carry_rehearsal[0]
                              else self.PHASE_NAMES[int(self._stack_phase[0])])
                print("[sequential-stack] env0 reset: phase={} step={} "
                      "timeout={} fall={}".format(
                          phase_name,
                          int(self.progress_buf[0]), timeout, fallen),
                      flush=True)
            self._sequential_reset_reported |= ended

    def post_physics_step(self):
        super().post_physics_step()
        # Full-stack mode in the parent leaves DONE alive until the global
        # timeout.  End this sequential episode immediately and normally once
        # the top box has remained stable for the requested number of steps.
        self.reset_buf[self._stack_phase == self.DONE] = 1
        # A1-only curriculum: A2_RESUME means A1 successfully placed,
        # released (or exhausted the bounded release grace), and retreated.
        # End here so early training spends no rollout budget on A2.  Full
        # sequential behavior remains the environment default.
        if self.stack_end_on_a2_resume:
            self.reset_buf[self._stack_phase == self.A2_RESUME] = 1
        self.extras["stack_carry_rehearsal"] = \
            self._carry_rehearsal.repeat_interleave(self.num_agents)
        if self._is_eval:
            # The inherited success buffer means ordinary carry delivery.
            # Sequential-stack evaluation instead succeeds only after the
            # top box has passed VERIFY_STACK and reached DONE.
            stack_success = (self._stack_phase == self.DONE)
            self.extras["success"] = stack_success.repeat_interleave(
                self.num_agents)

    def update_metrics(self):
        """Accumulate path error separately for the three sequence stages."""
        super().update_metrics()
        # update_metrics can run during super().__init__, before these buffers
        # are allocated.  The ordinary MA-steer columns are still collected.
        if not hasattr(self, "_seq_ep_steps"):
            return

        rows = torch.arange(self._rows, device=self.device)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        phase = self._stack_phase[env]
        rehearsal = self._carry_rehearsal[env]
        groups = (
            (phase == self.A1_PLACE) | (phase == self.VERIFY_BOTTOM),
            phase == self.A1_RETREAT,
            (phase == self.A2_RESUME) | (phase == self.VERIFY_STACK),
        )
        for group, mask in enumerate(groups):
            mask = mask & ~rehearsal
            self._seq_ep_lat_root[mask, group] += self._lat_root[mask].abs()
            self._seq_ep_lat_box[mask, group] += self._lat_box[mask].abs()
            self._seq_ep_steps[mask, group] += 1

    def _metric_extra_cols(self, rows):
        """Append sequential metrics after the inherited fixed columns.

        Columns 40..50 are success, final phase, then root/box cumulative
        lateral error and step count for place, retreat and A2 respectively.
        Columns 51..53 record the physical box XYZ size for stratified eval.
        """
        cols = super()._metric_extra_cols(rows)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        success = (self._stack_phase[env] == self.DONE).float()
        phase = self._stack_phase[env].float()
        cols.extend((success, phase))
        for group in range(3):
            cols.extend((
                self._seq_ep_lat_root[rows, group],
                self._seq_ep_lat_box[rows, group],
                self._seq_ep_steps[rows, group].float(),
            ))
        cols.extend((self._box_lib._box_size[rows, axis]
                     for axis in range(3)))
        return cols

    def _metric_reset_extra(self, rows):
        super()._metric_reset_extra(rows)
        if hasattr(self, "_seq_ep_steps"):
            self._seq_ep_lat_root[rows] = 0.0
            self._seq_ep_lat_box[rows] = 0.0
            self._seq_ep_steps[rows] = 0

    def _hand_box_clearance(self, rows):
        """Minimum hand-center distance from the owned oriented-box surface."""
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands = rigid[:, self._key_body_ids[[0, 1]]]
        boxes = self.humanoid_rows(self._box_states)[rows]
        sizes = self._box_lib._box_size[rows]
        return self._body_to_box_clearance(hands, boxes, sizes)

    def _foot_box_clearance(self, rows):
        """Minimum foot-center clearance from the owned oriented box."""
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        feet = rigid[:, self._key_body_ids[[2, 3]]]
        boxes = self.humanoid_rows(self._box_states)[rows]
        sizes = self._box_lib._box_size[rows]
        return self._body_to_box_clearance(feet, boxes, sizes)

    def _clearance_score(self, clearance):
        span = self.stack_hand_clear_done - self.stack_hand_clear_start
        return ((clearance - self.stack_hand_clear_start) / span).clamp(0.0, 1.0)

    def _compute_reward(self, actions):
        """Keep the ms18 task reward and shape only valid box release events."""
        # Save the unmodified MA-steer carry reward for rehearsal envs.  The
        # stack parent necessarily computes its coordination reward for the
        # whole vectorized batch, so restore these rows after stack shaping.
        rehearsal_reward = None
        if (self._carry_rehearsal is not None
                and bool(self._carry_rehearsal.any())):
            HumanoidMASteerCarry._compute_reward(self, actions)
            rehearsal_rows = self.agent_rows(torch.nonzero(
                self._carry_rehearsal, as_tuple=False).squeeze(-1))
            rehearsal_reward = self.rew_buf[rehearsal_rows].clone()
        super()._compute_reward(actions)
        if not hasattr(self, "_stack_phase"):
            return

        rows = self.all_rows().view(self.num_envs, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        roots = self.humanoid_rows(self._humanoid_root_states)
        boxes = self.humanoid_rows(self._box_states)
        phase = self._stack_phase

        # A1: releasing is useful only after the bottom box is accurately
        # placed and nearly stationary.  This prevents a throw-and-step-away
        # shortcut while supplying a dense gradient for moving both hands off.
        bottom_delta = boxes[r0, 0:3] - self._bottom_nominal_pos
        bottom_xy = torch.norm(bottom_delta[:, 0:2], dim=-1)
        bottom_speed2 = (boxes[r0, 7:10] ** 2).sum(dim=-1)
        bottom_ang2 = (boxes[r0, 10:13] ** 2).sum(dim=-1)
        # All A1 placement/release shaping is XY-only.  Height is checked by
        # the coordinator's 3-D placement condition, not by this reward.
        bottom_position_quality = torch.exp(-12.0 * bottom_xy ** 2)
        bottom_quality = (bottom_position_quality
                          * torch.exp(-4.0 * bottom_speed2
                                      - 0.5 * bottom_ang2))
        a1_clear = self._clearance_score(self._hand_box_clearance(r0))
        verifying = phase == self.VERIFY_BOTTOM
        # Keep the native XYZ term during A1_PLACE: its Z component is the
        # dense signal that actually lowers the box.  Only after the box has
        # reached the strict VERIFY_BOTTOM height band replace that 0.4
        # contribution (2 * carry_r's 0.2 * pos_near) with XY distance.
        native_pos_xyz = torch.exp(-10.0 * (bottom_delta ** 2).sum(dim=-1))
        native_pos_xy = torch.exp(-10.0 * bottom_xy ** 2)
        self.rew_buf[r0] += verifying.float() * 0.4 * (
            native_pos_xy - native_pos_xyz)

        # A1_PLACE keeps the native grasp incentive.  After entering
        # VERIFY_BOTTOM, remove only its +0.2 handheld term; position,
        # steering and the rest of carry reward remain intact.
        native_handheld = self._handheld_score(
            roots[r0, 0:3], boxes[r0, 0:3],
            self.humanoid_rows(self._rigid_body_pos)[r0])
        self.rew_buf[r0] -= verifying.float() * 0.2 * native_handheld
        a1_valid_release = bottom_quality * a1_clear
        a1_early_release = (1.0 - bottom_quality) * a1_clear
        # Do not multiply hands-on by stability: lifting/moving the box must
        # not make the penalty disappear while the hands remain attached.
        a1_hands_on = bottom_position_quality * (1.0 - a1_clear)
        self.rew_buf[r0] += verifying.float() * (
            self.stack_release_reward_w * a1_valid_release
            - self.stack_early_release_penalty * a1_early_release
            - self.stack_hands_on_penalty * a1_hands_on)
        if self.stack_foot_box_penalty > 0.0:
            foot_clear = self._foot_box_clearance(r0)
            foot_overlap = ((self.stack_foot_box_clearance - foot_clear)
                            / max(self.stack_foot_box_clearance, 1e-4))
            foot_overlap = foot_overlap.clamp(0.0, 1.0)
            # Penalize only the unsafe band.  Beyond the required clearance
            # there is no extra reward for moving the box farther away.
            self.rew_buf[r0] -= (verifying.float()
                                 * self.stack_foot_box_penalty
                                 * foot_overlap)

        # A2: preserve grasp while approaching, then reverse the incentive at
        # the committed top pose.  Exact centering, height, stillness and hand
        # separation must all agree; proximity alone cannot earn this term.
        top_delta = boxes[r1, 0:3] - self._committed_top_pos
        top_xy = torch.norm(top_delta[:, 0:2], dim=-1)
        top_z = torch.abs(top_delta[:, 2])
        top_speed2 = (boxes[r1, 7:10] ** 2).sum(dim=-1)
        top_ang2 = (boxes[r1, 10:13] ** 2).sum(dim=-1)
        top_quality = (torch.exp(-50.0 * top_xy ** 2)
                       * torch.exp(-40.0 * top_z ** 2)
                       * torch.exp(-4.0 * top_speed2 - 0.5 * top_ang2))
        a2_clear = self._clearance_score(self._hand_box_clearance(r1))
        stacking = ((phase == self.A2_RESUME)
                    | (phase == self.VERIFY_STACK))
        a2_valid_release = top_quality * a2_clear
        a2_early_release = (1.0 - top_quality) * a2_clear
        # Near the target, cancel the parent's residual preference for keeping
        # hands attached and make a clean release strictly more profitable.
        hands_still_on = top_quality * (1.0 - a2_clear)
        self.rew_buf[r1] += stacking.float() * (
            0.25 * top_quality
            + self.stack_release_reward_w * a2_valid_release
            - self.stack_early_release_penalty * a2_early_release
            - self.stack_hands_on_penalty * hands_still_on)
        if rehearsal_reward is not None:
            self.rew_buf[rehearsal_rows] = rehearsal_reward

    def _virtual_retreat_carry_obs(self, rows, env_ids):
        """Place a virtual pickup box at A1's retreat endpoint."""
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        real_box = self.humanoid_rows(self._box_states)[rows]
        heading_inv = torch_utils.calc_heading_quat_inv(roots[:, 3:7])

        # The physical box remains at the placement target.  In observation
        # space, put a stationary duplicate on the floor at the retreat
        # endpoint so carry asks A1 to approach/pick it up, not to keep holding
        # the box it just released while walking away.
        box_pos = self._a1_retreat_pos[env_ids].clone()
        box_pos[:, 2] = real_box[:, 2]
        box_rot = real_box[:, 3:7]
        box_vel = torch.zeros_like(roots[:, 7:10])
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

        # Box and placement goal coincide: this is the native "walk to box"
        # part of carry, with no request to transport an already-held object.
        carry_goal = box_pos
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
        # Native carry rehearsal must bypass every stack-specific target and
        # object rewrite.  Both episode types still share the exact 340-D ABI.
        if self._carry_rehearsal is None:
            super()._post_object_reset(env_ids)
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long)
            carry_ids = env_ids[self._carry_rehearsal[env_ids]]
            stack_ids = env_ids[~self._carry_rehearsal[env_ids]]
            HumanoidMAStackCarry._post_object_reset(self, stack_ids)
            HumanoidMASteerCarry._post_object_reset(self, carry_ids)
            if len(carry_ids) > 0:
                # No normal stack phase compares equal to -1, so coordinator,
                # virtual retreat box and early curriculum termination remain
                # inactive for native carry episodes.
                self._stack_phase[carry_ids] = -1
                self._stack_phase_age[carry_ids] = 0
            env_ids = stack_ids
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

        bottom_delta = boxes[r0, 0:3] - self._bottom_nominal_pos
        bottom_xy_err = torch.norm(bottom_delta[:, 0:2], dim=-1)
        bottom_z_err = torch.abs(bottom_delta[:, 2])
        bottom_slow = (
            (torch.norm(boxes[r0, 7:10], dim=-1) < self.stack_vel_tol)
            & (torch.norm(boxes[r0, 10:13], dim=-1)
               < self.stack_ang_vel_tol))
        foot_clear = self._foot_box_clearance(r0)
        feet_safe = foot_clear >= self.stack_foot_box_clearance
        bottom_at_target = ((bottom_xy_err < self.stack_pos_tol)
                            & (bottom_z_err < self.stack_bottom_z_tol))
        bottom_candidate = bottom_at_target & bottom_slow & feet_safe
        self._bottom_stable_count = torch.where(
            bottom_candidate, self._bottom_stable_count + 1,
            torch.zeros_like(self._bottom_stable_count))

        phase = self._stack_phase
        verify = torch.nonzero(
            (phase == self.A1_PLACE) & bottom_at_target,
            as_tuple=False).squeeze(-1)
        self._set_phase(verify, self.VERIFY_BOTTOM)

        phase = self._stack_phase
        lost_bottom = torch.nonzero(
            (phase == self.VERIFY_BOTTOM) & ~bottom_at_target,
            as_tuple=False).squeeze(-1)
        self._set_phase(lost_bottom, self.A1_PLACE)

        phase = self._stack_phase
        bottom_hands_clear = (
            self._hand_box_clearance(r0) >= self.stack_hand_clear_done)
        # Give the release reward a bounded training window.  A hard hand gate
        # trapped episodes forever, while removing it advanced to retreat as
        # soon as the box settled and provided almost no release experience.
        # Clear hands can advance immediately after physical stabilization;
        # otherwise the coordinator waits only up to the grace timeout.
        release_ready = (bottom_hands_clear
                         | (self._stack_phase_age
                            >= self.stack_release_grace_steps))
        placed = torch.nonzero(
            (phase == self.VERIFY_BOTTOM)
            & (self._bottom_stable_count >= self.stack_stable_steps)
            & release_ready,
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
