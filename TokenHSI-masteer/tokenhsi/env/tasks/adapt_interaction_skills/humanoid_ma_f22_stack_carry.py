"""Rule-coordinated two-agent box stacking on top of the frozen f22 task.

This class owns only the shared-world contract.  Agent rows still receive the
validated f22 observation (dual steer + carry); a separate coordination block
is prepended for the trainable coordination token.

The coordinator is deliberately state based:

  A1 carry/place bottom -> verify stable -> retreat
  A2 carry to staging   -> hold          -> place top

The top goal is committed from the *measured* bottom-box pose after it has
remained slow and close to the nominal bottom goal for several frames.
"""

import os

import torch

from isaacgym import gymtorch
from env.tasks.adapt_interaction_skills.humanoid_ma_f22_steer_carry import (
    HumanoidMAF22SteerCarry,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_carry import CARRY_HI, CARRY_LO
from tokenhsi.utils import steer_path as sp
from utils import torch_utils
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate


def _f(name, default):
    return float(os.environ.get(name, default))


class HumanoidMAF22StackCarry(HumanoidMAF22SteerCarry):
    """A=2 stacking environment with an explicit phase coordinator."""

    # phase ids are part of the policy contract; append rather than reorder.
    A1_PLACE = 0
    VERIFY_BOTTOM = 1
    A1_RETREAT = 2
    A2_RESUME = 3
    VERIFY_STACK = 4
    DONE = 5
    NUM_PHASES = 6

    TASK_FULL = 0
    TASK_HOLD = 1
    TASK_RETREAT = 2
    TASK_TOP = 3
    NUM_CURRICULUM_TASKS = 4

    # role(2) + phase(6) + teammate root(4) + teammate box(4) +
    # committed top goal(4) + bottom velocity(6) + flags(4)
    COORD_DIM = 30

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 2)) != 2:
            raise ValueError("HumanoidMAF22StackCarry currently requires numAgents=2")
        self.stack_task_mode = os.environ.get("STACK_TASK_MODE", "stack")
        if self.stack_task_mode not in ("stack", "putdown_retreat"):
            raise ValueError("STACK_TASK_MODE must be stack or putdown_retreat")
        self.stack_pos_tol = _f("STACK_POS_TOL", 0.18)
        self.stack_top_xy_tol = _f("STACK_TOP_XY_TOL", 0.08)
        self.stack_top_z_tol = _f("STACK_TOP_Z_TOL", 0.08)
        self.stack_vel_tol = _f("STACK_VEL_TOL", 0.12)
        self.stack_ang_vel_tol = _f("STACK_ANG_VEL_TOL", 0.35)
        self.stack_stable_steps = int(_f("STACK_STABLE_STEPS", 20))
        # The isolated putdown diagnostic gives frozen steer enough runway to
        # turn away from the box and then walk forward.  Full stacking keeps
        # its shorter hand-off clearance unless explicitly overridden.
        retreat_default = 3.0 if self.stack_task_mode == "putdown_retreat" else 1.5
        self.stack_retreat_dist = _f("STACK_RETREAT_DIST", retreat_default)
        self.stack_top_clearance = _f("STACK_TOP_CLEARANCE", 0.02)
        self.stack_hold_radius = _f("STACK_HOLD_RADIUS", 0.45)
        self.stack_phase_timeout = int(_f("STACK_PHASE_TIMEOUT", 240))
        self.stack_coord_reward_w = _f("STACK_COORD_REWARD_W", 0.25)
        self.stack_retreat_base_reward_w = _f("STACK_RETREAT_BASE_REWARD_W", 0.0)
        self.stack_retreat_time_penalty = _f("STACK_RETREAT_TIME_PENALTY", 0.03)
        self.stack_retreat_success_bonus = _f("STACK_RETREAT_SUCCESS_BONUS", 1.0)
        self.stack_retreat_progress_w = _f("STACK_RETREAT_PROGRESS_W", 0.60)
        self.stack_retreat_velocity_w = _f("STACK_RETREAT_VELOCITY_W", 0.25)
        self.stack_retreat_target_speed = _f("STACK_RETREAT_TARGET_SPEED", 0.80)
        self.stack_retreat_heading_w = _f("STACK_RETREAT_HEADING_W", 0.25)
        self.stack_wait_base_reward_w = _f("STACK_WAIT_BASE_REWARD_W", 0.0)
        self.stack_body_clearance = _f("STACK_BODY_CLEARANCE", 0.22)
        self.stack_cross_box_clearance = _f("STACK_CROSS_BOX_CLEARANCE", 0.22)
        self.stack_own_box_clearance = _f("STACK_OWN_BOX_CLEARANCE", 0.20)
        self.stack_own_box_penalty = _f("STACK_OWN_BOX_PENALTY", 0.75)
        self.stack_stand_penalty = _f("STACK_STAND_PENALTY", 0.25)
        self.stack_collision_penalty = _f("STACK_COLLISION_PENALTY", 0.35)
        self.stack_curriculum = bool(int(os.environ.get("STACK_CURRICULUM", "0")))
        self.stack_curr_stage1_frames = int(_f("STACK_CURR_STAGE1_FRAMES", 0))
        self.stack_curr_stage2_frames = int(_f("STACK_CURR_STAGE2_FRAMES", 0))
        self._curriculum_stage = 0 if self.stack_curriculum else 2
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        E = self.num_envs
        self._stack_phase = torch.zeros(E, dtype=torch.long, device=self.device)
        self._stack_phase_age = torch.zeros(E, dtype=torch.long, device=self.device)
        self._bottom_stable_count = torch.zeros(E, dtype=torch.long, device=self.device)
        self._top_stable_count = torch.zeros(E, dtype=torch.long, device=self.device)
        self._top_committed = torch.zeros(E, dtype=torch.bool, device=self.device)
        self._committed_top_pos = torch.zeros(E, 3, device=self.device)
        self._committed_top_yaw = torch.zeros(E, device=self.device)
        self._bottom_nominal_pos = torch.zeros(E, 3, device=self.device)
        self._a2_stage_pos = torch.zeros(E, 3, device=self.device)
        self._a1_retreat_pos = torch.zeros(E, 3, device=self.device)
        self._a1_retreat_start = torch.zeros(E, 2, device=self.device)
        self._a1_retreat_dir = torch.zeros(E, 2, device=self.device)
        R = E * 2
        self._ever_held = torch.zeros(R, dtype=torch.bool, device=self.device)
        self._drop_penalized = torch.zeros(R, dtype=torch.bool, device=self.device)
        self._initial_box_bottom_z = torch.zeros(R, device=self.device)
        self._committed_bottom_pos = torch.zeros(E, 3, device=self.device)
        self._prev_stage_dist = torch.zeros(E, device=self.device)
        self._prev_retreat_dist = torch.zeros(E, device=self.device)
        self._prev_top_dist = torch.zeros(E, device=self.device)
        self._coord_reward = torch.zeros(R, device=self.device)
        self._held = torch.zeros(R, dtype=torch.bool, device=self.device)
        self._collision_cost = torch.zeros(E, device=self.device)
        self._own_box_collision_cost = torch.zeros(E, device=self.device)
        self._retreat_lateral = torch.zeros(E, device=self.device)
        self._retreat_along = torch.zeros(E, device=self.device)
        self._retreat_heading_align = torch.zeros(E, device=self.device)
        self._curriculum_task = torch.zeros(E, dtype=torch.long, device=self.device)

    @staticmethod
    def _body_to_box_clearance(body_pos, box_state, box_size):
        """Minimum rigid-body-center distance to an oriented box surface."""
        E, B, _ = body_pos.shape
        delta = body_pos - box_state[:, None, 0:3]
        inv = quat_conjugate(box_state[:, 3:7])
        inv = inv[:, None, :].expand(-1, B, -1)
        local = quat_rotate(inv.reshape(-1, 4), delta.reshape(-1, 3)).view(E, B, 3)
        outside = (torch.abs(local) - 0.5 * box_size[:, None, :]).clamp(min=0.0)
        return torch.norm(outside, dim=-1).amin(dim=-1)

    def set_train_info(self, env_frames, *args, **kwargs):
        if not self.stack_curriculum or self.stack_task_mode != "stack":
            return
        if env_frames < self.stack_curr_stage1_frames:
            stage = 0
        elif env_frames < self.stack_curr_stage2_frames:
            stage = 1
        else:
            stage = 2
        if stage != self._curriculum_stage:
            self._curriculum_stage = stage
            print("[stack curriculum] task-mixture stage={} frame={}".format(
                stage, env_frames), flush=True)

    def get_task_obs_size(self):
        return self.COORD_DIM + self.steer_dim() + (CARRY_HI - CARRY_LO)

    def get_multi_task_info(self):
        if getattr(self, "_use_base_info", False):
            return super().get_multi_task_info()
        each = [self.COORD_DIM, self.steer_dim(), CARRY_HI - CARRY_LO,
                CARRY_HI - CARRY_LO]
        names = ["new_extra_coord", "old_steer", "new_carry", "old_carry"]
        index = torch.cumsum(torch.tensor([0] + each), dim=0).to(self.device)
        mask = torch.zeros(len(each), sum(each), dtype=torch.bool, device=self.device)
        for i in range(len(each)):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": len(each),
            "tota_subtask_obs_size": sum(each),
            "each_subtask_obs_size": each,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": False,
            "each_subtask_name": names,
            "major_task_name": "carry",
            "has_extra": True,
        }

    def _reset_envs(self, env_ids):
        """Reset the shared env and commit both humanoid poses afterwards."""
        if len(env_ids) == 0 or not hasattr(self, "_humanoid_root_states"):
            return super()._reset_envs(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_envs(env_ids)

        # The parent commits humanoid tensors before the stack-specific object
        # hook runs.  Commit both humanoids once more at the very end so a new
        # phase/task can never be exposed while the simulator still owns the
        # preceding episode's ragdoll root or DOF state.
        actor_ids = self._humanoid_actor_ids_per_env[env_ids].reshape(-1)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def _reset_task_indicator(self, env_ids):
        """Assign phase-curriculum tasks and valid carryWith reference starts."""
        super()._reset_task_indicator(env_ids)
        if not hasattr(self, "_curriculum_task") or len(env_ids) == 0:
            return
        if self.stack_task_mode == "putdown_retreat":
            self._curriculum_task[env_ids] = self.TASK_FULL
            # Keep the parent's normal pickup/carry reset distribution.  This
            # mode changes only what happens after placement; it must not turn
            # the episode into a synthetic box-in-hands diagnostic reset.
            return
        else:
            if not self.stack_curriculum:
                self._curriculum_task[env_ids] = self.TASK_FULL
                return
            if self._curriculum_stage == 0:
                probs = [0.50, 0.50, 0.00, 0.00]
            elif self._curriculum_stage == 1:
                probs = [0.25, 0.25, 0.25, 0.25]
            else:
                probs = [0.50, 0.15, 0.15, 0.20]
            prob = torch.tensor(probs, device=self.device)
            mode = torch.multinomial(prob, len(env_ids), replacement=True)
            self._curriculum_task[env_ids] = mode
            forced = env_ids[mode != self.TASK_FULL]

        # Later-phase tasks need a physically valid box-in-hands reset. Replace
        # whichever carry skill the parent sampled with carryWith, keeping the
        # env and row-indexed motion arrays aligned.
        if len(forced) == 0:
            return
        carry_env = self._reset_ref_env_ids.get("carry", {})
        carry_mid = self._reset_ref_motion_ids.get("carry", {})
        carry_time = self._reset_ref_motion_times.get("carry", {})
        A = self.num_agents
        for skill_name in list(carry_env.keys()):
            skill_envs = carry_env[skill_name]
            remove = (skill_envs[:, None] == forced[None, :]).any(dim=1)
            keep = ~remove
            carry_env[skill_name] = skill_envs[keep]
            keep_rows = keep.repeat_interleave(A)
            carry_mid[skill_name] = carry_mid[skill_name][keep_rows]
            carry_time[skill_name] = carry_time[skill_name][keep_rows]

        motion_lib = self._motion_lib["carryWith"]
        motion_ids = motion_lib.sample_motions(len(forced) * A)
        motion_times = motion_lib.sample_time_rsi(motion_ids)
        carry_env["carryWith"] = torch.cat((
            carry_env.get("carryWith", forced[:0]), forced), dim=0)
        carry_mid["carryWith"] = torch.cat((
            carry_mid.get("carryWith", motion_ids[:0]), motion_ids), dim=0)
        carry_time["carryWith"] = torch.cat((
            carry_time.get("carryWith", motion_times[:0]), motion_times), dim=0)

    def _post_object_reset(self, env_ids):
        # The first reset happens inside the parent constructor, before our
        # stack buffers exist.  Keep that construction path unchanged.
        super()._post_object_reset(env_ids)
        if not hasattr(self, "_stack_phase") or len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids).view(-1, 2)
        b = self.humanoid_rows(self._box_states)
        h = self.humanoid_rows(self._humanoid_root_states)
        rigid = self.humanoid_rows(self._rigid_body_pos)
        r0, r1 = rows[:, 0], rows[:, 1]

        # A1's reset target is the nominal bottom placement.  A2 waits on the
        # side of the stack opposite its current position.
        bottom = self._box_tar_pos[r0].clone()
        away = h[r1, 0:2] - bottom[:, 0:2]
        away = away / torch.norm(away, dim=-1, keepdim=True).clamp(min=1e-4)
        stage = bottom.clone()
        stage[:, 0:2] += away * self.stack_retreat_dist
        # Before the dependency opens, A2 sees a reachable hold pose rather
        # than the future top placement.  Mean hand height keeps this target
        # off the floor, so the frozen carry actor is asked to retain the box.
        hands = rigid[r1][:, self._key_body_ids[[0, 1]]].mean(dim=1)
        stage[:, 2] = hands[:, 2]
        # Initialized from A1's transformed state below.  A2's position must
        # not determine which way A1 steps back.
        retreat = h[r0, 0:3].clone()

        mode = self._curriculum_task[env_ids]
        if self.stack_task_mode == "putdown_retreat":
            # Preserve A1's original pickup/carry reset exactly.  Only isolate
            # the inactive A2 by translating its humanoid and box together, so
            # their reference hand-box relation is unchanged and A2 cannot
            # collide with A1 during this single-agent diagnostic.
            desired_a2_box_xy = bottom[:, 0:2] + 3.0 * away
            shift_a2 = desired_a2_box_xy - b[r1, 0:2]
            h[r1, 0:2] += shift_a2
            b[r1, 0:2] += shift_a2
            stage = b[r1, 0:3].clone()

            actor_ids = torch.cat((
                self._humanoid_actor_ids_per_env[env_ids, 1],
                self._box_actor_ids[r1]), dim=0)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            self._refresh_sim_tensors()

        phase_reset = mode != self.TASK_FULL
        if phase_reset.any():
            # carryWith reset gives a valid hand-box relation. Translate A2 and
            # its box together to the requested subtask start without breaking
            # that relation.
            hold_ids = torch.nonzero(
                (mode == self.TASK_HOLD) | (mode == self.TASK_RETREAT),
                as_tuple=False).squeeze(-1)
            if len(hold_ids) > 0:
                rr = r1[hold_ids]
                shift = stage[hold_ids, 0:2] - b[rr, 0:2]
                h[rr, 0:2] += shift
                b[rr, 0:2] += shift

            top_ids = torch.nonzero(mode == self.TASK_TOP,
                                    as_tuple=False).squeeze(-1)
            if len(top_ids) > 0:
                rr = r1[top_ids]
                size = self._box_lib._box_size
                top_start = bottom[top_ids, 0:2] + away[top_ids] * 0.60
                shift = top_start - b[rr, 0:2]
                h[rr, 0:2] += shift
                b[rr, 0:2] += shift

            placed_ids = torch.nonzero(
                (mode == self.TASK_RETREAT) | (mode == self.TASK_TOP),
                as_tuple=False).squeeze(-1)
            if len(placed_ids) > 0:
                br = r0[placed_ids]
                b[br, 0:3] = bottom[placed_ids]
                b[br, 7:13] = 0.0
                root_dist = torch.where(
                    (mode[placed_ids] == self.TASK_TOP).unsqueeze(-1),
                    torch.full((len(placed_ids), 1), self.stack_retreat_dist,
                               device=self.device),
                    torch.full((len(placed_ids), 1), 0.40, device=self.device))
                h[br, 0:2] = bottom[placed_ids, 0:2] - away[placed_ids] * root_dist
                h[br, 7:13] = 0.0

            changed_envs = env_ids[phase_reset]
            changed_rows = self.agent_rows(changed_envs)
            actor_ids = torch.cat((
                self._humanoid_actor_ids_per_env[changed_envs].reshape(-1),
                self._box_actor_ids[changed_rows]), dim=0)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            self._refresh_sim_tensors()
        self._stack_phase[env_ids] = self.A1_PLACE
        self._stack_phase_age[env_ids] = 0
        self._bottom_stable_count[env_ids] = 0
        self._top_stable_count[env_ids] = 0
        self._top_committed[env_ids] = False
        self._bottom_nominal_pos[env_ids] = bottom
        self._a2_stage_pos[env_ids] = stage
        self._a1_retreat_pos[env_ids] = retreat
        self._update_retreat_goal(env_ids)
        self._committed_top_pos[env_ids] = stage
        self._committed_top_yaw[env_ids] = 0.0
        self._committed_bottom_pos[env_ids] = bottom

        self._box_tar_pos[r0] = bottom
        # Do not expose the future top goal before A1's measured bottom pose is
        # stable.  Both carry observation and steering point to the hold pose.
        self._box_tar_pos[r1] = stage
        self._reset_steer_to(r0, bottom)
        self._reset_steer_to(r1, stage)

        # A2's carry target is a box-relative goal, not a physical platform.
        # Hide only its blue target-platform actor; A1's bottom plate remains.
        if self._carry_reset_random_height:
            tar_platform = self.agent_axis(self._tar_platform_pos)
            tar_platform[env_ids, 1, 2] = -10.0
            actor_ids = self._tar_platform_actor_ids[r1]
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

        rr = rows.reshape(-1)
        self._ever_held[rr] = False
        self._drop_penalized[rr] = False
        self._initial_box_bottom_z[rr] = (
            b[rr, 2] - 0.5 * self._box_lib._box_size[rr, 2])
        self._prev_stage_dist[env_ids] = torch.norm(
            b[r1, 0:2] - stage[:, 0:2], dim=-1)
        self._prev_retreat_dist[env_ids] = torch.norm(
            h[r0, 0:2] - retreat[:, 0:2], dim=-1)
        self._prev_top_dist[env_ids] = torch.norm(
            b[r1, 0:3] - stage, dim=-1)
        self._coord_reward[rr] = 0.0
        self._held[rr] = False

        # Initialize later-phase latches consistently with their reference
        # state. A2 starts with a carried box; A1 in retreat/top starts after
        # having already carried and released the bottom box.
        forced_ids = torch.nonzero(mode != self.TASK_FULL,
                                   as_tuple=False).squeeze(-1)
        if len(forced_ids) > 0:
            a2r = r1[forced_ids]
            self._ever_held[a2r] = True
            self._held[a2r] = True
            self._initial_box_bottom_z[a2r] -= 0.10
        a1_done_ids = torch.nonzero(
            (mode == self.TASK_RETREAT) | (mode == self.TASK_TOP),
            as_tuple=False).squeeze(-1)
        if len(a1_done_ids) > 0:
            self._ever_held[r0[a1_done_ids]] = True

        retreat_ids = torch.nonzero(mode == self.TASK_RETREAT,
                                    as_tuple=False).squeeze(-1)
        top_ids = torch.nonzero(mode == self.TASK_TOP,
                                as_tuple=False).squeeze(-1)
        committed = torch.cat((retreat_ids, top_ids), dim=0)
        if len(committed) > 0:
            self._commit_top_goal(env_ids[committed])
        if len(retreat_ids) > 0:
            ids = env_ids[retreat_ids]
            self._set_phase(ids, self.A1_RETREAT)
            self._activate_retreat_steer(ids)
            self._reset_steer_to(r1[retreat_ids], self._a2_stage_pos[ids])
        if len(top_ids) > 0:
            ids = env_ids[top_ids]
            self._set_phase(ids, self.A2_RESUME)
            self._reset_steer_to(r1[top_ids], self._committed_top_pos[ids])

    def _reset_steer_to(self, rows, steer_goal):
        """Regenerate f22 steering without changing the carry placement goal."""
        if len(rows) == 0:
            return
        saved = self._box_tar_pos[rows].clone()
        try:
            self._box_tar_pos[rows] = steer_goal
            self._reset_steer(rows)
        finally:
            self._box_tar_pos[rows] = saved

    def _set_phase(self, env_ids, phase):
        if len(env_ids) == 0:
            return
        self._stack_phase[env_ids] = phase
        self._stack_phase_age[env_ids] = 0

    def _commit_top_goal(self, env_ids):
        rows = self.agent_rows(env_ids).view(-1, 2)
        r0, r1 = rows[:, 0], rows[:, 1]
        b = self.humanoid_rows(self._box_states)
        size = self._box_lib._box_size
        bottom = b[r0]
        top = bottom[:, 0:3].clone()
        top[:, 2] += 0.5 * (size[r0, 2] + size[r1, 2]) + self.stack_top_clearance
        self._committed_top_pos[env_ids] = top
        self._committed_bottom_pos[env_ids] = bottom[:, 0:3]
        q = bottom[:, 3:7]
        self._committed_top_yaw[env_ids] = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        self._top_committed[env_ids] = True
        self._box_tar_pos[r1] = top

    def _update_retreat_goal(self, env_ids):
        """Send A1 outward from the placed box, then let frozen steer turn."""
        if len(env_ids) == 0:
            return
        r0 = self.agent_rows(env_ids).view(-1, 2)[:, 0]
        h = self.humanoid_rows(self._humanoid_root_states)[r0]
        q = h[:, 3:7]
        heading = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        r0_box = self.humanoid_rows(self._box_states)[r0]
        outward = h[:, 0:2] - r0_box[:, 0:2]
        outward_norm = torch.norm(outward, dim=-1, keepdim=True)
        # At a normal placement A1 is behind the box, so this ray points away
        # from it and cannot cross the box.  Keep the old facing fallback only
        # for the degenerate case where both xy centers coincide.
        retreat_dir = torch.where(
            outward_norm > 1e-3,
            outward / outward_norm.clamp(min=1e-4),
            -forward)
        retreat = h[:, 0:3].clone()
        retreat[:, 0:2] += retreat_dir * self.stack_retreat_dist
        self._a1_retreat_start[env_ids] = h[:, 0:2]
        self._a1_retreat_dir[env_ids] = retreat_dir
        self._a1_retreat_pos[env_ids] = retreat

    def _activate_retreat_steer(self, env_ids):
        """Keep the parent steer path aligned with A1's outward command."""
        if len(env_ids) == 0:
            return
        r0 = self.agent_rows(env_ids).view(-1, 2)[:, 0]
        start = self._a1_retreat_start[env_ids]
        direction = self._a1_retreat_dir[env_ids]
        arc = torch.arange(sp.V, device=self.device).float() * sp.DS
        self._gt_path[r0] = (start[:, None, :]
                             + arc[None, :, None] * direction[:, None, :])
        self._s_end[r0] = self.stack_retreat_dist
        self._arc_root[r0] = 0.0
        self._arc_box[r0] = 0.0
        self._prev_arc[r0] = 0.0
        self._mscale[r0] = 1.0
        self._prev_retreat_dist[env_ids] = self.stack_retreat_dist
        if self.steer_endclamp:
            past_end = arc[None, :] > self.stack_retreat_dist
            self._mscale[r0] = torch.where(
                past_end, torch.zeros_like(self._mscale[r0]), self._mscale[r0])

    def _compute_reward(self, actions):
        """Preserve f22 reward and add bounded phase-specific coordination."""
        super()._compute_reward(actions)
        if not hasattr(self, "_stack_phase"):
            return

        E = self.num_envs
        rows = torch.arange(E, device=self.device)[:, None] * 2 + \
               torch.arange(2, device=self.device)[None, :]
        r0, r1 = rows[:, 0], rows[:, 1]
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        grasp_score = self._handheld_score(h[:, 0:3], b[:, 0:3])
        hands_on = grasp_score > 0.35
        box_bottom = b[:, 2] - 0.5 * self._box_lib._box_size[:, 2]
        lifted = box_bottom > (self._initial_box_bottom_z + 0.08)
        held = hands_on & lifted
        self._held.copy_(held)
        self._ever_held |= held

        phase = self._stack_phase
        coord = torch.zeros_like(self.rew_buf)
        retreat_only = self.stack_task_mode == "putdown_retreat"

        # A2: transport to staging, then learn to retain the grasp and stop.
        stage_dist = torch.norm(b[r1, 0:2] - self._a2_stage_pos[:, 0:2], dim=-1)
        stage_progress = (self._prev_stage_dist - stage_dist).clamp(-0.10, 0.10) / 0.10
        at_stage = stage_dist <= self.stack_hold_radius
        waiting = phase < self.A2_RESUME
        approaching = waiting & ~at_stage
        holding = waiting & at_stage
        if not retreat_only:
            coord[r1] += approaching.float() * (
                0.35 * stage_progress + 0.10 * grasp_score[r1])

        box_speed2 = (b[r1, 7:10] ** 2).sum(dim=-1)
        box_ang2 = (b[r1, 10:13] ** 2).sum(dim=-1)
        still_score = torch.exp(-4.0 * box_speed2 - 0.5 * box_ang2)
        stage_score = torch.exp(-2.0 * stage_dist ** 2)
        hold_score = grasp_score[r1] * lifted[r1].float()
        if not retreat_only:
            coord[r1] += holding.float() * (
                0.35 * hold_score + 0.15 * still_score + 0.10 * stage_score)

        dropped_now = (waiting & self._ever_held[r1] & ~held[r1]
                       & ~self._drop_penalized[r1])
        if not retreat_only:
            coord[r1] -= dropped_now.float()
        self._drop_penalized[r1] |= dropped_now

        # The first-drop event alone is too cheap: after paying it once, A2
        # could leave the box on the floor for the rest of WAIT.  Penalize the
        # invalid state continuously until the dependency opens or it regrips.
        dropped_wait = waiting & self._ever_held[r1] & ~held[r1]
        if not retreat_only:
            coord[r1] -= 0.35 * dropped_wait.float()

        # Before A2_RESUME the native carry target is the nominal top pose, so
        # its task reward directly encourages the early placement that WAIT is
        # meant to prevent.  The frozen actor is kept; only this conflicting
        # task reward is masked while coordination learns staging/hold.
        a2_base_reward = self.rew_buf[r1]
        if retreat_only:
            self.rew_buf[r1] = 0.0
        else:
            self.rew_buf[r1] = torch.where(
                waiting,
                self.stack_wait_base_reward_w * a2_base_reward,
                a2_base_reward)

        # A1: after placement, move the body away without disturbing/regrasping
        # the committed bottom box.
        retreat_dist = torch.norm(h[r0, 0:2] - self._a1_retreat_pos[:, 0:2], dim=-1)
        retreat_progress = (self._prev_retreat_dist - retreat_dist).clamp(-0.10, 0.10) / 0.10
        bottom_clear = torch.norm(h[r0, 0:2] - b[r0, 0:2], dim=-1)
        bottom_disp = torch.norm(b[r0, 0:3] - self._committed_bottom_pos, dim=-1)
        bottom_motion = ((b[r0, 7:10] ** 2).sum(dim=-1)
                         + 0.1 * (b[r0, 10:13] ** 2).sum(dim=-1))
        retreating = phase >= self.A1_RETREAT
        active_retreat = phase == self.A1_RETREAT
        retreat_success = bottom_clear >= self.stack_retreat_dist
        if retreat_only:
            # Project A1 onto the box-outward ray supplied to frozen steer and
            # penalize only perpendicular deviation from that safe path.
            rel = h[r0, 0:2] - self._a1_retreat_start
            along = (rel * self._a1_retreat_dir).sum(dim=-1)
            foot = self._a1_retreat_start + along.unsqueeze(-1) * self._a1_retreat_dir
            lateral = torch.norm(h[r0, 0:2] - foot, dim=-1)
            self._retreat_lateral.copy_(lateral)
            self._retreat_along.copy_(along)
            lat_pen = torch.exp(-0.5 * lateral ** 2) - 1.0
            along_vel = (h[r0, 7:9] * self._a1_retreat_dir).sum(dim=-1)
            q = h[r0, 3:7]
            heading = torch.atan2(
                2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            facing = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
            heading_align = (facing * self._a1_retreat_dir).sum(dim=-1).clamp(-1.0, 1.0)
            self._retreat_heading_align.copy_(heading_align)
            # Zero when aligned and negative otherwise. Unlike a positive
            # heading score, this cannot pay the policy for standing still.
            heading_penalty = heading_align - 1.0
            target_speed = self.stack_retreat_target_speed
            vel_at_zero = torch.exp(torch.tensor(
                -2.0 * target_speed ** 2, device=self.device))
            velocity_score = ((torch.exp(
                -2.0 * (target_speed - along_vel) ** 2) - vel_at_zero)
                / (1.0 - vel_at_zero).clamp(min=1e-4))
            nominal_root_z = self.humanoid_rows(
                self._initial_humanoid_root_states)[r0, 2]
            stand_deficit = ((nominal_root_z - h[r0, 2]) / 0.25).clamp(0.0, 1.0)
            # Dense steering terms provide a gradient all the way to the 3 m
            # terminal threshold. Both are exactly zero while stationary;
            # moving backwards makes progress negative and also lowers the
            # directional velocity score.
            coord[r0] += active_retreat.float() * (
                self.stack_retreat_progress_w * retreat_progress
                + self.stack_retreat_velocity_w * velocity_score
                + self.stack_retreat_heading_w * heading_penalty
                + 0.75 * lat_pen - 0.60 * hands_on[r0].float()
                - self.stack_stand_penalty * stand_deficit
                - 0.50 * bottom_disp.clamp(max=1.0)
                - 0.10 * bottom_motion.clamp(max=2.0)
                - self.stack_retreat_time_penalty
                + self.stack_retreat_success_bonus * retreat_success.float())
        else:
            coord[r0] += active_retreat.float() * (
                0.50 * retreat_progress
                - 0.50 * hands_on[r0].float()
                - 0.50 * bottom_disp.clamp(max=1.0)
                - 0.10 * bottom_motion.clamp(max=2.0)
                - self.stack_retreat_time_penalty
                + self.stack_retreat_success_bonus * retreat_success.float())

        # Native carry rewards continuously prefer staying close to the box,
        # which directly opposes A1's post-placement retreat objective.  Once
        # retreat starts, leave motion naturalness to AMP and train the task
        # behavior from the coordination reward instead.
        a1_base_reward = self.rew_buf[r0]
        self.rew_buf[r0] = torch.where(
            retreating,
            self.stack_retreat_base_reward_w * a1_base_reward,
            a1_base_reward)

        # A2 resume/top: distinguish horizontal centering from vertical height.
        # A single 3-D distance reward allowed visibly side-offset placements
        # to enter the old 0.18 m success ball.
        top_delta = b[r1, 0:3] - self._committed_top_pos
        top_xy_err = torch.norm(top_delta[:, 0:2], dim=-1)
        top_z_err = torch.abs(top_delta[:, 2])
        top_dist = torch.norm(top_delta, dim=-1)
        top_progress = (self._prev_top_dist - top_dist).clamp(-0.10, 0.10) / 0.10
        resuming = (phase >= self.A2_RESUME) & (phase < self.DONE)
        top_xy_score = torch.exp(-50.0 * top_xy_err ** 2)
        top_z_score = torch.exp(-40.0 * top_z_err ** 2)
        top_align_score = top_xy_score * top_z_score
        released = self._ever_held[r1] & ~held[r1]
        settled_score = top_align_score * still_score * released.float()
        lowered = top_delta[:, 2] < 0.12
        side_miss = ((top_xy_err - self.stack_top_xy_tol) /
                     max(self.stack_top_xy_tol, 1e-3)).clamp(0.0, 1.0)
        if not retreat_only:
            coord[r1] += resuming.float() * (
                0.20 * top_progress + 0.30 * top_align_score
                + 0.35 * settled_score + 0.10 * grasp_score[r1]
                - 0.45 * lowered.float() * side_miss
                - 0.40 * bottom_disp.clamp(max=1.0))

        # Collision geometry includes the whole humanoid and both cross-owned
        # boxes. Root distance alone misses the common case where a carried
        # box is wedged between two otherwise separated roots.
        body_pair_dist = self.agent_min_dist().view(E, 2)[:, 0]
        body_close = ((self.stack_body_clearance - body_pair_dist) /
                      max(self.stack_body_clearance, 1e-3)).clamp(0.0, 1.0)
        rigid = self._rigid_body_pos
        # Hands intentionally approach the bottom box while releasing the top
        # box. Exclude only those two bodies from cross-box clearance so normal
        # placement is not punished; body-body clearance still uses all links.
        non_hand = torch.ones(rigid.shape[2], dtype=torch.bool, device=self.device)
        non_hand[self._key_body_ids[[0, 1]]] = False
        cross_rigid = rigid[:, :, non_hand]
        size = self._box_lib._box_size
        a1_to_a2_box = self._body_to_box_clearance(cross_rigid[:, 0], b[r1], size[r1])
        a2_to_a1_box = self._body_to_box_clearance(cross_rigid[:, 1], b[r0], size[r0])
        cross_box_dist = torch.minimum(a1_to_a2_box, a2_to_a1_box)
        cross_box_close = ((self.stack_cross_box_clearance - cross_box_dist) /
                           max(self.stack_cross_box_clearance, 1e-3)).clamp(0.0, 1.0)
        # Once retreat begins, non-hand A1 links must clear its own placed box.
        # This catches a hip, leg or elbow sweeping through the box while the
        # root turns, before box displacement alone can report the damage.
        a1_own_box_dist = self._body_to_box_clearance(
            cross_rigid[:, 0], b[r0], size[r0])
        own_box_close = ((self.stack_own_box_clearance - a1_own_box_dist) /
                         max(self.stack_own_box_clearance, 1e-3)).clamp(0.0, 1.0)
        own_box_close = own_box_close * active_retreat.float()
        self._own_box_collision_cost.copy_(own_box_close)
        collision = torch.maximum(body_close, cross_box_close)
        if retreat_only:
            collision = own_box_close
            coord[r0] -= self.stack_own_box_penalty * own_box_close
        else:
            coord[r0] -= self.stack_collision_penalty * collision
            coord[r1] -= self.stack_collision_penalty * collision
            coord[r0] -= self.stack_own_box_penalty * own_box_close
        self._collision_cost.copy_(collision)

        coord = coord.clamp(-2.0, 1.0)
        self._coord_reward.copy_(coord)
        self.rew_buf += self.stack_coord_reward_w * coord
        self._prev_stage_dist.copy_(stage_dist)
        self._prev_retreat_dist.copy_(retreat_dist)
        self._prev_top_dist.copy_(top_dist)

    def _update_stack_coordinator(self):
        if not hasattr(self, "_stack_phase"):
            return
        self._stack_phase_age += 1
        rows = torch.arange(self.num_envs, device=self.device)[:, None] * 2 + \
               torch.arange(2, device=self.device)[None, :]
        r0, r1 = rows[:, 0], rows[:, 1]
        b = self.humanoid_rows(self._box_states)
        h = self.humanoid_rows(self._humanoid_root_states)

        bottom_err = torch.norm(b[r0, 0:3] - self._bottom_nominal_pos, dim=-1)
        bottom_slow = ((torch.norm(b[r0, 7:10], dim=-1) < self.stack_vel_tol) &
                       (torch.norm(b[r0, 10:13], dim=-1) < self.stack_ang_vel_tol))
        candidate = (bottom_err < self.stack_pos_tol) & bottom_slow
        self._bottom_stable_count = torch.where(
            candidate, self._bottom_stable_count + 1,
            torch.zeros_like(self._bottom_stable_count))

        p = self._stack_phase
        to_verify = torch.nonzero((p == self.A1_PLACE) &
                                  (bottom_err < self.stack_pos_tol), as_tuple=False).squeeze(-1)
        self._set_phase(to_verify, self.VERIFY_BOTTOM)
        stable = torch.nonzero((p == self.VERIFY_BOTTOM) &
                               (self._bottom_stable_count >= self.stack_stable_steps),
                               as_tuple=False).squeeze(-1)
        if len(stable) > 0:
            self._commit_top_goal(stable)
            self._update_retreat_goal(stable)
            self._activate_retreat_steer(stable)
            self._set_phase(stable, self.A1_RETREAT)

        clear = torch.norm(h[r0, 0:2] - b[r0, 0:2], dim=-1) >= self.stack_retreat_dist
        resume = torch.nonzero((p == self.A1_RETREAT) & clear, as_tuple=False).squeeze(-1)
        if len(resume) > 0 and self.stack_task_mode == "stack":
            self._reset_steer_to(resume * 2 + 1, self._committed_top_pos[resume])
        if self.stack_task_mode == "putdown_retreat":
            self._set_phase(resume, self.DONE)
        else:
            self._set_phase(resume, self.A2_RESUME)

        top_delta = b[r1, 0:3] - self._committed_top_pos
        top_xy_err = torch.norm(top_delta[:, 0:2], dim=-1)
        top_z_err = torch.abs(top_delta[:, 2])
        top_slow = ((torch.norm(b[r1, 7:10], dim=-1) < self.stack_vel_tol) &
                    (torch.norm(b[r1, 10:13], dim=-1) < self.stack_ang_vel_tol))
        top_candidate = ((top_xy_err < self.stack_top_xy_tol) &
                         (top_z_err < self.stack_top_z_tol) & top_slow)
        self._top_stable_count = torch.where(
            top_candidate, self._top_stable_count + 1,
            torch.zeros_like(self._top_stable_count))
        verify = torch.nonzero((p == self.A2_RESUME) &
                               (top_xy_err < self.stack_top_xy_tol) &
                               (top_z_err < self.stack_top_z_tol),
                               as_tuple=False).squeeze(-1)
        self._set_phase(verify, self.VERIFY_STACK)
        lost = torch.nonzero((p == self.VERIFY_STACK) & ~top_candidate,
                             as_tuple=False).squeeze(-1)
        self._set_phase(lost, self.A2_RESUME)
        done = torch.nonzero((p == self.VERIFY_STACK) &
                             (self._top_stable_count >= self.stack_stable_steps),
                             as_tuple=False).squeeze(-1)
        self._set_phase(done, self.DONE)

        # A timeout does not invent a success transition.  It exposes a flag
        # to the token and leaves the normal episode timeout/reset in charge.

    def _dual_steer_obs(self, rows):
        """Phase commands separated from the immutable box placement goals.

        A2 gets a shrinking vector to the staging anchor while waiting.  A1's
        root gets a retreat command after placement while its box half of the
        dual window remains pinned to the bottom target.
        """
        out = super()._dual_steer_obs(rows)
        if not hasattr(self, "_stack_phase") or self.steer_zero:
            return out
        e, a = torch.div(rows, 2, rounding_mode="floor"), rows % 2
        phase = self._stack_phase[e]
        # Do not replace the approach/carry window before A2 has actually
        # transported its box to staging.  The parent path contains the vital
        # root->box leg used by f22 to initiate pickup.
        b = self.humanoid_rows(self._box_states)[rows]
        at_stage = (torch.norm(b[:, 0:2] - self._a2_stage_pos[e, 0:2], dim=-1)
                    <= self.stack_hold_radius)
        wait = (a == 1) & (phase < self.A2_RESUME) & at_stage
        retreat = (a == 0) & (phase >= self.A1_RETREAT)
        special = wait | retreat
        if not special.any():
            return out

        sr = rows[special]
        se, sa = e[special], a[special]
        h = self.humanoid_rows(self._humanoid_root_states)[sr]
        root_goal = torch.where(
            (sa == 0).unsqueeze(-1), self._a1_retreat_pos[se, 0:2],
            self._a2_stage_pos[se, 0:2])
        box_goal = torch.where(
            (sa == 0).unsqueeze(-1), self._bottom_nominal_pos[se, 0:2],
            self._a2_stage_pos[se, 0:2])
        pts = torch.cat((
            root_goal[:, None, :].expand(-1, self.steer_k, -1),
            box_goal[:, None, :].expand(-1, self.steer_k, -1),
        ), dim=1)
        retreat_special = sa == 0
        if retreat_special.any():
            re = se[retreat_special]
            rh = h[retreat_special, 0:2]
            direction = self._a1_retreat_dir[re]
            along = ((rh - self._a1_retreat_start[re]) * direction).sum(dim=-1)
            along = along.clamp(min=0.0)
            if self.steer_clip:
                along = along.clamp(max=self.stack_retreat_dist)
            off = (torch.arange(1, self.steer_k + 1, device=self.device).float()
                   / self.steer_k)
            sample_s = along[:, None] + self.steer_m_nom * off[None, :]
            if self.steer_clip:
                sample_s = sample_s.clamp(max=self.stack_retreat_dist)
            retreat_pts = (self._a1_retreat_start[re, None, :]
                           + sample_s[..., None] * direction[:, None, :])
            pts[retreat_special, :self.steer_k] = retreat_pts
        heading = torch.atan2(
            2.0 * (h[:, 6] * h[:, 5] + h[:, 3] * h[:, 4]),
            1.0 - 2.0 * (h[:, 4] ** 2 + h[:, 5] ** 2))
        out[special] = sp.to_local(pts, h[:, 0:2], heading).reshape(len(sr), -1)
        return out

    def _coord_obs(self, rows):
        A = self.num_agents
        e, a = torch.div(rows, A, rounding_mode="floor"), rows % A
        other = e * A + (1 - a)
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        inv = torch_utils.calc_heading_quat_inv(h[rows, 3:7])

        def local(v):
            return torch.clamp(quat_rotate(inv, v), -10.0, 10.0)

        role = torch.nn.functional.one_hot(a, 2).float()
        phase = torch.nn.functional.one_hot(self._stack_phase[e], self.NUM_PHASES).float()
        root_rel = local(h[other, 0:3] - h[rows, 0:3])
        root_yaw = torch_utils.quat_to_tan_norm(quat_mul(inv, h[other, 3:7]))[:, :1]
        box_rel = local(b[other, 0:3] - h[rows, 0:3])
        box_yaw = torch_utils.quat_to_tan_norm(quat_mul(inv, b[other, 3:7]))[:, :1]
        top_rel = local(self._committed_top_pos[e] - h[rows, 0:3])
        q = h[rows, 3:7]
        ego_yaw = torch.atan2(2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                              1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        top_yaw = torch.atan2(torch.sin(self._committed_top_yaw[e] - ego_yaw),
                              torch.cos(self._committed_top_yaw[e] - ego_yaw)).unsqueeze(-1)
        bottom_vel = torch.cat((b[e * A, 7:10], b[e * A, 10:13]), dim=-1)
        own_box = b[rows]
        own_slow = ((torch.norm(own_box[:, 7:10], dim=-1) < self.stack_vel_tol) &
                    (torch.norm(own_box[:, 10:13], dim=-1) < self.stack_ang_vel_tol))
        bottom_err = torch.norm(
            own_box[:, 0:3] - self._bottom_nominal_pos[e], dim=-1)
        top_delta = own_box[:, 0:3] - self._committed_top_pos[e]
        top_xy_err = torch.norm(top_delta[:, 0:2], dim=-1)
        top_z_err = torch.abs(top_delta[:, 2])
        a1_release_ready = ((a == 0) & own_slow &
                            (bottom_err < self.stack_pos_tol))
        a2_release_ready = ((a == 1) &
                            (self._stack_phase[e] >= self.A2_RESUME) &
                            own_slow &
                            (top_xy_err < self.stack_top_xy_tol) &
                            (top_z_err < self.stack_top_z_tol))
        release_ready = a1_release_ready | a2_release_ready
        flags = torch.stack((
            self._top_committed[e].float(),
            self._held[rows].float(),
            self._ever_held[rows].float(),
            release_ready.float(),
        ), dim=-1)
        out = torch.cat((role, phase, root_rel, root_yaw, box_rel, box_yaw,
                         top_rel, top_yaw, bottom_vel, flags), dim=-1)
        assert out.shape[-1] == self.COORD_DIM, out.shape
        return out

    def _compute_task_obs(self, env_ids=None):
        f22 = super()._compute_task_obs(env_ids)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        # Base construction computes observations before the coordinator
        # buffers can be allocated.  That first observation is discarded by
        # the normal initial reset; preserve its declared shape meanwhile.
        if not hasattr(self, "_stack_phase"):
            coord = torch.zeros(len(rows), self.COORD_DIM, device=self.device)
        else:
            coord = self._coord_obs(rows)
        return torch.cat((coord, f22), dim=-1)

    def _compute_amp_observations(self, env_ids=None):
        """Use the locomotion prior for A1 after its box has been placed."""
        super()._compute_amp_observations(env_ids)
        if (not self._enable_task_specific_disc or
                not hasattr(self, "_stack_phase")):
            return

        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        agent = rows % self.num_agents
        retreat = ((agent == 0) &
                   (self._stack_phase[env] == self.A1_RETREAT))
        retreat_rows = rows[retreat]
        if len(retreat_rows) == 0:
            return

        # Only the task-conditioning suffix of the AMP discriminator input is
        # changed. Actor observations remain carry-conditioned, while the
        # released A1 motion is judged against the generic traj/loco prior.
        labels = torch.zeros((len(retreat_rows), self._num_tasks),
                             device=self.device,
                             dtype=self._curr_amp_obs_buf.dtype)
        labels[:, self.TaskUID["traj"].value] = 1.0
        self._curr_amp_obs_buf[retreat_rows, -self._num_tasks:] = labels

    def post_physics_step(self):
        super().post_physics_step()
        # The parent has refreshed simulator tensors.  Update commands from
        # that state, then replace only the next policy observation.
        self._update_stack_coordinator()
        if self.stack_task_mode == "putdown_retreat":
            # This diagnostic task ends at clearance; do not keep collecting
            # released/clearance state reward until the global timeout.
            self.reset_buf[self._stack_phase == self.DONE] = 1
        self._compute_observations()
        if hasattr(self, "extras"):
            self.extras["policy_obs"] = self.obs_buf.clone()
            self.extras["stack_phase"] = self._stack_phase.repeat_interleave(2)
            self.extras["stack_done"] = (self._stack_phase == self.DONE).repeat_interleave(2)
            self.extras["stack_held"] = self._held
            self.extras["stack_coord_reward"] = self._coord_reward
            # Raw environment/task reward before AMP mixing.  The viewer uses
            # this together with disc_reward to display the exact PPO reward.
            self.extras["stack_task_reward"] = self.rew_buf.clone()
            self.extras["stack_collision_cost"] = self._collision_cost.repeat_interleave(2)
            self.extras["stack_own_box_collision_cost"] = \
                self._own_box_collision_cost.repeat_interleave(2)
            self.extras["stack_retreat_lateral"] = \
                self._retreat_lateral.repeat_interleave(2)
            self.extras["stack_retreat_along"] = \
                self._retreat_along.repeat_interleave(2)
            self.extras["stack_retreat_heading_align"] = \
                self._retreat_heading_align.repeat_interleave(2)
            self.extras["stack_curriculum_stage"] = torch.full(
                (self.num_envs * 2,), self._curriculum_stage,
                dtype=torch.long, device=self.device)
            self.extras["stack_curriculum_task"] = \
                self._curriculum_task.repeat_interleave(2)
