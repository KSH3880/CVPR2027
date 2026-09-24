"""Carry/no-box steering with a flag-free mid-path stop curriculum.

The actor ABI remains 340-D. STOP is represented only by the geometry of the
existing six 2-D steering points: they contract to one fixed world-space anchor
and later expand along the same path. BRAKE/HOLD/RESUME are controller-side
bookkeeping and are never appended to policy observations.
"""

import os

import torch
from isaacgym.torch_utils import quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_carry_steer_mix import (
    HumanoidMACarrySteerMix,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp
from tokenhsi.utils.steer_stop import contract_to_anchor, cosine_transition


def _f(name, default):
    return float(os.environ.get(name, default))


class HumanoidMACarrySteerStopMix(HumanoidMACarrySteerMix):
    """Four-way mixture: carry/no-box x uninterrupted/stop-resume."""

    WAIT, BRAKE, HOLD, RESUME, DONE = range(5)

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.stop_prob = _f("MS_STOP_PROB", 0.5)
        self.stop_brake_t = _f("MS_STOP_BRAKE_T", 0.8)
        self.stop_resume_t = _f("MS_STOP_RESUME_T", 0.8)
        self.stop_hold_min = _f("MS_STOP_HOLD_MIN", 0.5)
        self.stop_hold_max = _f("MS_STOP_HOLD_MAX", 2.0)
        self.stop_margin = _f("MS_STOP_MARGIN", 0.8)
        self.stop_post_box = _f("MS_STOP_POST_BOX", 0.8)
        self.stop_brake_dist = _f("MS_STOP_BRAKE_DIST", 0.6)
        self.stop_reward_w = _f("MS_STOP_REWARD_W", 1.0)
        if not 0.0 <= self.stop_prob <= 1.0:
            raise ValueError("MS_STOP_PROB must be in [0, 1]")
        if self.stop_brake_t <= 0 or self.stop_resume_t <= 0:
            raise ValueError("MS_STOP_BRAKE_T and MS_STOP_RESUME_T must be positive")
        if self.stop_hold_min < 0 or self.stop_hold_min > self.stop_hold_max:
            raise ValueError("invalid MS_STOP_HOLD_MIN/MAX")
        if min(self.stop_margin, self.stop_post_box, self.stop_brake_dist) < 0:
            raise ValueError("stop distances must be non-negative")
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.stop_brake_steps = max(1, round(self.stop_brake_t / self.dt))
        self.stop_resume_steps = max(1, round(self.stop_resume_t / self.dt))

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        R = self._rows
        self._stop_enabled = torch.zeros(R, dtype=torch.bool, device=self.device)
        self._stop_phase = torch.full((R,), self.DONE, dtype=torch.long, device=self.device)
        self._stop_anchor_s = torch.zeros(R, device=self.device)
        self._stop_trigger_s = torch.zeros(R, device=self.device)
        self._stop_anchor_xy = torch.zeros(R, 2, device=self.device)
        self._stop_blend = torch.zeros(R, device=self.device)
        self._stop_start_step = torch.full((R,), -1, dtype=torch.long, device=self.device)
        self._stop_hold_steps = torch.zeros(R, dtype=torch.long, device=self.device)
        self._stop_root_height = torch.zeros(R, device=self.device)
        self._stop_last_update = torch.full((R,), -1, dtype=torch.long, device=self.device)

        self._ep_stop_cmd_n = torch.zeros(R, device=self.device)
        self._ep_stop_hold_n = torch.zeros(R, device=self.device)
        self._ep_stop_root_v = torch.zeros(R, device=self.device)
        self._ep_stop_box_v = torch.zeros(R, device=self.device)
        self._ep_stop_anchor_err = torch.zeros(R, device=self.device)
        self._ep_stop_height_err = torch.zeros(R, device=self.device)
        self._ep_stop_held = torch.zeros(R, device=self.device)
        self._ep_stop_resumed = torch.zeros(R, device=self.device)

    def _points_at(self, rows, arc):
        q = (arc / sp.DS).clamp(0, sp.V - 2)
        lo = q.floor().long()
        frac = (q - lo.float())[:, None]
        path = self._gt_path[rows]
        ar = torch.arange(len(rows), device=self.device)
        return path[ar, lo] + frac * (path[ar, lo + 1] - path[ar, lo])

    def _post_object_reset(self, env_ids):
        # The mixed parent first creates carry paths and then replaces the
        # no-box half with standalone 3 m paths. Sample anchors only afterward.
        super()._post_object_reset(env_ids)
        self._reset_stop(self.agent_rows(env_ids))

    def _reset_stop(self, rows):
        if len(rows) == 0:
            return
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        solo = self._solo_env[env]
        lo = torch.where(
            solo,
            torch.full((len(rows),), self.stop_margin, device=self.device),
            self._arc_box[rows] + self.stop_post_box,
        )
        hi = self._s_end[rows] - self.stop_margin
        valid = hi > lo + 0.2

        seed = self.steer_seed + self._steer_tick + int(rows[0]) + 8101
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        choose = torch.rand(len(rows), generator=gen).to(self.device) < self.stop_prob
        where = torch.rand(len(rows), generator=gen).to(self.device)
        hold = self.stop_hold_min + (
            self.stop_hold_max - self.stop_hold_min
        ) * torch.rand(len(rows), generator=gen).to(self.device)

        enabled = choose & valid
        anchor_s = lo + where * (hi - lo).clamp(min=0.0)
        self._stop_enabled[rows] = enabled
        self._stop_phase[rows] = torch.where(
            enabled,
            torch.full_like(rows, self.WAIT),
            torch.full_like(rows, self.DONE),
        )
        self._stop_anchor_s[rows] = anchor_s
        self._stop_trigger_s[rows] = (anchor_s - self.stop_brake_dist).clamp(min=0.0)
        self._stop_anchor_xy[rows] = self._points_at(rows, anchor_s)
        self._stop_blend[rows] = 0.0
        self._stop_start_step[rows] = -1
        self._stop_hold_steps[rows] = torch.clamp((hold / self.dt).round().long(), min=1)
        roots = self.humanoid_rows(self._humanoid_root_states)
        self._stop_root_height[rows] = roots[rows, 2]
        self._stop_last_update[rows] = -1

    def _carry_is_held(self, rows):
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        box = self.humanoid_rows(self._box_states)[rows, :3]
        hands = rigid[:, self._key_body_ids[[0, 1]]]
        hand_near = torch.norm(hands - box[:, None, :], dim=-1).max(dim=-1).values < 0.75
        root = self.humanoid_rows(self._humanoid_root_states)[rows, :3]
        box_near = torch.norm(box[:, :2] - root[:, :2], dim=-1) < 0.8
        lifted = box[:, 2] > 0.35
        return hand_near & box_near & lifted

    def _update_stop_command(self, rows):
        if len(rows) == 0:
            return
        step = self.progress_rows()[rows]
        fresh = self._stop_last_update[rows] != step
        rows = rows[fresh]
        step = step[fresh]
        if len(rows) == 0:
            return
        self._stop_last_update[rows] = step

        roots = self.humanoid_rows(self._humanoid_root_states)
        # Base _ratchet() projects against every row in self._gt_path. Reset
        # observations can contain only the just-reset envs, so project against
        # the matching subset explicitly (e.g. 6 reset rows, not all 2048).
        prev_arc = self._arc_root[rows]
        lo = prev_arc - self.steer_back
        hi = prev_arc + 2.0
        arc, _, _ = sp.project(roots[rows, :2], self._gt_path[rows], lo, hi)
        arc = torch.maximum(arc, lo)
        if self.steer_clip:
            arc = torch.minimum(arc, self._s_end[rows])
        phase = self._stop_phase[rows]
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        solo = self._solo_env[env]
        held = solo | self._carry_is_held(rows)
        start = (
            self._stop_enabled[rows]
            & (phase == self.WAIT)
            & (arc >= self._stop_trigger_s[rows])
            & held
        )
        if start.any():
            rr = rows[start]
            self._stop_phase[rr] = self.BRAKE
            self._stop_start_step[rr] = step[start]
            self._stop_root_height[rr] = roots[rr, 2]

        phase = self._stop_phase[rows]
        brake = phase == self.BRAKE
        if brake.any():
            rr = rows[brake]
            elapsed = step[brake] - self._stop_start_step[rr]
            self._stop_blend[rr] = cosine_transition(elapsed, self.stop_brake_steps, True)
            done = elapsed >= self.stop_brake_steps
            if done.any():
                self._stop_phase[rr[done]] = self.HOLD
                self._stop_blend[rr[done]] = 1.0

        phase = self._stop_phase[rows]
        hold_mask = phase == self.HOLD
        if hold_mask.any():
            rr = rows[hold_mask]
            elapsed = step[hold_mask] - self._stop_start_step[rr] - self.stop_brake_steps
            done = elapsed >= self._stop_hold_steps[rr]
            if done.any():
                rd = rr[done]
                self._stop_phase[rd] = self.RESUME
                self._stop_start_step[rd] = step[hold_mask][done]

        phase = self._stop_phase[rows]
        resume = phase == self.RESUME
        if resume.any():
            rr = rows[resume]
            elapsed = step[resume] - self._stop_start_step[rr]
            self._stop_blend[rr] = cosine_transition(elapsed, self.stop_resume_steps, False)
            done = elapsed >= self.stop_resume_steps
            if done.any():
                rd = rr[done]
                self._stop_phase[rd] = self.DONE
                self._stop_blend[rd] = 0.0

    def _compute_task_obs(self, env_ids=None):
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        self._update_stop_command(rows)
        return super()._compute_task_obs(env_ids)

    def _m_at(self, arc, rows=None):
        base = super()._m_at(arc, rows)
        if not hasattr(self, "_stop_blend"):
            return base
        ar = torch.arange(len(arc), device=self.device) if rows is None else rows
        return base * (1.0 - self._stop_blend[ar])

    def _steer_world_points(self, rows):
        # Use raw M here. Contraction itself reduces inter-point spacing, while
        # _m_at exposes that same reduced spacing to the reward.
        arc = self._arc_root[rows]
        raw_m = HumanoidMASteerCarry._m_at(self, arc, rows)
        off = torch.arange(1, self.steer_k + 1, device=self.device)[None, :] / self.steer_k
        pts_arc = arc[:, None] + raw_m[:, None] * off
        if self.steer_clip:
            pts_arc = torch.minimum(pts_arc, self._s_end[rows][:, None])
        q = (pts_arc / sp.DS).clamp(0, sp.V - 2)
        lo = q.floor().long()
        frac = (q - lo.float())[..., None]
        ar = torch.arange(len(rows), device=self.device)[:, None]
        path = self._gt_path[rows]
        points = path[ar, lo] + frac * (path[ar, lo + 1] - path[ar, lo])
        return contract_to_anchor(points, self._stop_anchor_xy[rows], self._stop_blend[rows])

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        blend = self._stop_blend
        active = blend > 0
        if not active.any():
            return
        rows = torch.where(active)[0]
        h = self.humanoid_rows(self._humanoid_root_states)[rows]
        b = self.humanoid_rows(self._box_states)[rows]
        root_v = torch.norm(h[:, 7:9], dim=-1)
        root_w = torch.norm(h[:, 10:13], dim=-1)
        anchor_err = torch.norm(h[:, :2] - self._stop_anchor_xy[rows], dim=-1)
        height_err = torch.abs(h[:, 2] - self._stop_root_height[rows])
        up = torch.zeros(len(rows), 3, device=self.device)
        up[:, 2] = 1.0
        upright = quat_rotate(h[:, 3:7], up)[:, 2].clamp(min=0.0)
        quality = (
            0.8 * torch.exp(-4.0 * root_v.square())
            + 0.25 * torch.exp(-2.0 * root_w.square())
            + 0.5 * torch.exp(-5.0 * anchor_err.square())
            + 0.35 * torch.exp(-10.0 * height_err.square())
            + 0.35 * upright.square()
        )

        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        carry = ~self._solo_env[env]
        if carry.any():
            rc = rows[carry]
            box_v = torch.norm(b[carry, 7:10], dim=-1)
            box_w = torch.norm(b[carry, 10:13], dim=-1)
            rigid = self.humanoid_rows(self._rigid_body_pos)[rc]
            hands = rigid[:, self._key_body_ids[[0, 1]]]
            hand_err = torch.norm(hands - b[carry, None, :3], dim=-1).mean(dim=-1)
            quality[carry] += (
                0.6 * torch.exp(-4.0 * box_v.square())
                + 0.2 * torch.exp(-2.0 * box_w.square())
                + 0.5 * torch.exp(-5.0 * hand_err.square())
            )
        self.rew_buf[rows] += self.stop_reward_w * blend[rows] * quality

    def update_metrics(self):
        super().update_metrics()
        active = self._stop_blend > 0.01
        hold = self._stop_phase == self.HOLD
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        root_v = torch.norm(h[:, 7:9], dim=-1)
        box_v = torch.norm(b[:, 7:10], dim=-1)
        anchor_err = torch.norm(h[:, :2] - self._stop_anchor_xy, dim=-1)
        height_err = torch.abs(h[:, 2] - self._stop_root_height)
        rows = self.all_rows()
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        carry = ~self._solo_env[env]
        held = torch.ones(self._rows, device=self.device)
        if carry.any():
            held[carry] = self._carry_is_held(rows[carry]).float()
        self._ep_stop_cmd_n += active.float()
        self._ep_stop_hold_n += hold.float()
        self._ep_stop_root_v += torch.where(hold, root_v, torch.zeros_like(root_v))
        self._ep_stop_box_v += torch.where(hold & carry, box_v, torch.zeros_like(box_v))
        self._ep_stop_anchor_err += torch.where(hold, anchor_err, torch.zeros_like(anchor_err))
        self._ep_stop_height_err += torch.where(hold, height_err, torch.zeros_like(height_err))
        self._ep_stop_held += torch.where(hold & carry, held, torch.zeros_like(held))
        self._ep_stop_resumed = torch.maximum(
            self._ep_stop_resumed,
            (self._stop_phase == self.DONE).float() * self._stop_enabled.float(),
        )

    def _metric_extra_cols(self, rows):
        cols = super()._metric_extra_cols(rows)
        cols += [
            self._stop_enabled[rows].float(),
            self._ep_stop_cmd_n[rows],
            self._ep_stop_hold_n[rows],
            self._ep_stop_root_v[rows],
            self._ep_stop_box_v[rows],
            self._ep_stop_anchor_err[rows],
            self._ep_stop_height_err[rows],
            self._ep_stop_held[rows],
            self._ep_stop_resumed[rows],
        ]
        return cols

    def _metric_reset_extra(self, rows):
        super()._metric_reset_extra(rows)
        for value in (
            self._ep_stop_cmd_n,
            self._ep_stop_hold_n,
            self._ep_stop_root_v,
            self._ep_stop_box_v,
            self._ep_stop_anchor_err,
            self._ep_stop_height_err,
            self._ep_stop_held,
            self._ep_stop_resumed,
        ):
            value[rows] = 0.0
