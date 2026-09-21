"""Juan sequential-stack scoring on the local stack-release environment."""

import os

import numpy as np
import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_sequential_stack_release import (
    HumanoidMASequentialStackRelease,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)


class HumanoidMAJuanEvalSequentialStackRelease(
        HumanoidMASequentialStackRelease):
    """Keep the local scenario and expose Juan's evaluation contract."""

    TERM_NONE = 0
    TERM_FALL_A1 = 1
    TERM_FALL_A2 = 2
    TERM_FALL_BOTH = 3
    TERM_BOTTOM_DISPLACED = 4
    TERM_MULTIPLE = 5
    TERM_UNKNOWN = 6

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        horizon = int(os.environ.get("STACK_EPISODE_LENGTH", "900"))
        cfg["env"]["episodeLength"] = horizon
        cfg["env"]["episodeLengthShort"] = horizon
        self._juan_success_mode = os.environ.get(
            "STACK_EVAL_SUCCESS_MODE", "box_radius").strip()
        if self._juan_success_mode not in (
                "strict_done", "box_radius", "tokenhsi_carry"):
            raise ValueError(
                "STACK_EVAL_SUCCESS_MODE must be strict_done, box_radius, "
                "or tokenhsi_carry")
        self._juan_bottom_displace_tol = float(os.environ.get(
            "STACK_BOTTOM_DISPLACE_TOL", "0.50"))
        super().__init__(cfg, sim_params, physics_engine, device_type,
                         device_id, headless)
        self._juan_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._juan_terminate_reason = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._juan_ep_lat_root = torch.zeros((self._rows, 3), device=self.device)
        self._juan_ep_lat_box = torch.zeros((self._rows, 3), device=self.device)
        self._juan_ep_steps = torch.zeros(
            (self._rows, 3), dtype=torch.long, device=self.device)

    def _load_box_asset(self, box_sizes):
        if int(os.environ.get("STACK_EVAL_BOX_GRID", "1")) == 0:
            return super()._load_box_asset(box_sizes)

        ids = [int(value) for value in os.environ.get(
            "STACK_EVAL_BOX_SIZE_IDS", "0,4,7").split(",")]
        if len(ids) != 3 or len(set(ids)) != 3:
            raise ValueError("STACK_EVAL_BOX_SIZE_IDS must contain 3 unique ids")
        pool = np.asarray(self._box_lib._build_test_sizes, dtype=np.float32)
        if min(ids) < 0 or max(ids) >= len(pool):
            raise ValueError("STACK_EVAL_BOX_SIZE_IDS is outside testSizes")
        envs = box_sizes.shape[0] // 2
        if envs % 9:
            raise ValueError("3x3 box grid requires numEnvs divisible by 9")

        selected = pool[ids]
        sampled = np.empty((envs, 2, 3), dtype=np.float32)
        for env in range(envs):
            bottom, top = divmod(env % 9, 3)
            sampled[env, 0] = selected[bottom]
            sampled[env, 1] = selected[top]
        box_sizes[:] = torch.as_tensor(
            sampled.reshape(-1, 3), device=box_sizes.device)
        self._box_lib._build_box_bps()
        self._ss_base_agent_np = np.zeros(envs, dtype=np.int64)
        return HumanoidMASteerCarry._load_box_asset(self, box_sizes)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if hasattr(self, "_juan_success"):
            self._juan_success[env_ids] = False

    def _compute_reset(self):
        super()._compute_reset()
        if not hasattr(self, "_juan_terminate_reason"):
            return
        contact = self.humanoid_rows(self._contact_forces).clone()
        contact[:, self._contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(contact) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)
        body_height = self.humanoid_rows(self._rigid_body_pos)[..., 2]
        fall_height = body_height < self._termination_heights
        fall_height[:, self._contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        fallen = (fall_contact & fall_height
                  & (self.progress_rows() > 1)).view(self.num_envs, 2)
        fall_code = (fallen[:, 0].long() * self.TERM_FALL_A1
                     + fallen[:, 1].long() * self.TERM_FALL_A2)
        terminated = self._terminate_buf.bool()
        self._juan_terminate_reason[terminated] = torch.where(
            fall_code[terminated] > 0,
            fall_code[terminated],
            torch.full_like(fall_code[terminated], self.TERM_UNKNOWN))

    def update_metrics(self):
        super().update_metrics()
        if not hasattr(self, "_juan_ep_steps"):
            return
        rows = self.all_rows()
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        phase = self._ss_phase[env]
        groups = (
            (phase == self.CARRY) | (phase == self.RELEASE),
            phase == self.CLEAR,
            phase == self.STACK,
        )
        for group, mask in enumerate(groups):
            self._juan_ep_lat_root[mask, group] += self._lat_root[mask].abs()
            self._juan_ep_lat_box[mask, group] += self._lat_box[mask].abs()
            self._juan_ep_steps[mask, group] += 1

    def _metric_extra_cols(self, rows):
        cols = HumanoidMASteerCarry._metric_extra_cols(self, rows)
        if not hasattr(self, "_juan_ep_steps"):
            return cols
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        cols.extend((self._juan_success[env].float(), self._ss_phase[env].float()))
        for group in range(3):
            cols.extend((
                self._juan_ep_lat_root[rows, group],
                self._juan_ep_lat_box[rows, group],
                self._juan_ep_steps[rows, group].float(),
            ))
        cols.extend(self._box_lib._box_size[rows, axis] for axis in range(3))
        cols.append(self._juan_terminate_reason[env].float())
        return cols

    def _metric_reset_extra(self, rows):
        HumanoidMASteerCarry._metric_reset_extra(self, rows)
        if not hasattr(self, "_juan_ep_steps"):
            return
        self._juan_ep_lat_root[rows] = 0.0
        self._juan_ep_lat_box[rows] = 0.0
        self._juan_ep_steps[rows] = 0
        env = torch.unique(torch.div(
            rows, self.num_agents, rounding_mode="floor"))
        self._juan_terminate_reason[env] = self.TERM_NONE

    def post_physics_step(self):
        super().post_physics_step()
        if not hasattr(self, "_juan_success"):
            return
        env = torch.arange(self.num_envs, device=self.device)
        base_rows, top_rows = self._role_rows(env)
        boxes = self.humanoid_rows(self._box_states)
        phase = self._ss_phase
        active = phase == self.STACK
        top_delta = boxes[top_rows, 0:3] - self._ss_top_goal
        top_z_ok = top_delta[:, 2].abs() < self._ss_top_z_tol
        if self._juan_success_mode == "strict_done":
            reached = self._ss_success
        elif self._juan_success_mode == "tokenhsi_carry":
            reached = active & (
                torch.norm(top_delta, dim=-1) <= self._success_threshold)
        else:
            support_radius = 0.5 * torch.norm(
                self._box_lib._box_size[base_rows, 0:2], dim=-1)
            top_xy = torch.norm(
                boxes[top_rows, 0:2] - boxes[base_rows, 0:2], dim=-1)
            reached = active & (top_xy <= support_radius) & top_z_ok
        self._juan_success[reached] = True

        monitor = (phase >= self.CLEAR) & ~self._ss_rehearsal
        bottom_disp = torch.norm(
            boxes[base_rows, 0:3] - self._ss_latched_base, dim=-1)
        displaced = monitor & (bottom_disp > self._juan_bottom_displace_tol)
        prior = self._juan_terminate_reason[displaced]
        self._juan_terminate_reason[displaced] = torch.where(
            prior == self.TERM_NONE,
            torch.full_like(prior, self.TERM_BOTTOM_DISPLACED),
            torch.full_like(prior, self.TERM_MULTIPLE))
        self._juan_success[displaced] = False
        self.reset_buf[displaced] = 1
        self._terminate_buf[displaced] = 1

        self.reset_buf[self._ss_phase == self.SUCCESS] = 1
        if self._is_eval:
            self.extras["success"] = self._juan_success.repeat_interleave(2)
        self.extras["terminate"] = self._terminate_buf.repeat_interleave(2)
