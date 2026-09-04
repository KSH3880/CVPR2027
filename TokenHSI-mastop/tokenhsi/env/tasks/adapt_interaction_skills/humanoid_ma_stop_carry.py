"""Carry-aware stop curriculum built on MASteer."""

import os
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import HumanoidMASteerCarry, _f
from env.tasks.adapt_interaction_skills.humanoid_ma_carry import TEAMMATE_DIM
from tokenhsi.utils import steer_path as sp

A, B, C, D = range(4)
STAGES = {"A": 0, "B": 1, "C": 2, "D": 3}
MIXES = {"A": "1,0,0,0", "B": ".5,.5,0,0",
         "C": ".35,.25,.40,0", "D": ".35,.20,.20,.25"}


def _mix(value, stage):
    p = [float(x) for x in value.split(",")]
    if len(p) != 4 or any(x < 0 for x in p):
        raise ValueError("MSTOP_MIX must contain four non-negative values")
    if any(p[i] for i in range(stage + 1, 4)):
        raise ValueError("MSTOP_MIX contains a later stage")
    total = sum(p)
    if total <= 0:
        raise ValueError("empty MSTOP_MIX")
    return [x / total for x in p]


class HumanoidMAStopCarry(HumanoidMASteerCarry):

    def _scen_env(self):
        os.environ["MS_SCEN"] = "free"
        os.environ.pop("MA_LAYOUT", None)
        os.environ.setdefault("MA_SEP", "0")
        os.environ.setdefault("MA_SPAWN_GAP", "1.0")
        return "free"

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.stop_stage = os.environ.get("MSTOP_STAGE", "A").upper()
        if self.stop_stage not in STAGES:
            raise ValueError("MSTOP_STAGE must be A, B, C, or D")
        self.stop_stage_i = STAGES[self.stop_stage]
        self.stop_mix = _mix(
            os.environ.get("MSTOP_MIX", MIXES[self.stop_stage]), self.stop_stage_i)
        self.stop_seed = int(_f("MSTOP_SEED", 0))
        self.stop_prob = _f("MSTOP_STOP_PROB", 1.0)
        self.stop_len = _f("MSTOP_L", 12.0)
        self.stop_solo_gap = _f("MSTOP_SOLO_GAP", 9.0)
        self.stop_after_pick = _f("MSTOP_SOLO_AFTER", 1.5)
        self.stop_hold_min = _f("MSTOP_HOLD_MIN", .75)
        self.stop_hold_max = _f("MSTOP_HOLD_MAX", 1.5)
        self.stop_brake_time = _f("MSTOP_BRAKE_T", .8)
        self.stop_resume_time = _f("MSTOP_RESUME_T", .8)
        self.stop_track_w = _f("MSTOP_TRACK_W", .5)
        self.stop_track_delta = _f("MSTOP_TRACK_DELTA", .5)
        self.stop_carrywith = bool(int(_f("MSTOP_CARRYWITH", 0)))
        # Opt-in diagnostic: retain A's free layout/path for scenario B while
        # keeping B's stop schedule. Defaults preserve existing A/B behavior.
        self.stop_a_layout = bool(int(_f("MSTOP_A_LAYOUT", 0)))
        self.stop_floor_start = bool(int(_f("MS_FLOOR_START", 0)))
        self.stop_brake = _f("MSTOP_BRAKE_D", 1.2)
        self.stop_clear = _f("MSTOP_CLEAR", 1.0)
        self.stop_horizon = _f("MSTOP_HORIZON", 3.0)
        self.stop_alpha = _f("MSTOP_SPEED_EMA", .2)
        self.stop_resume_v = _f("MSTOP_RESUME_SPEED", .5)
        self.stop_speed = _f("MSTOP_STOP_SPEED", .2)
        self.stop_root_w = _f("MSTOP_ROOT_W", .5)
        self.stop_box_w = _f("MSTOP_BOX_W", .5)
        self.stop_hold_w = _f("MSTOP_HOLD_W", .5)
        self.stop_creep_c = _f("MSTOP_CREEP_C", .2)
        self.stop_drop_c = _f("MSTOP_DROP_C", 1.0)
        self.stop_collision_c = _f("MSTOP_COLLISION_C", 1.0)
        self.stop_hold_strict_w = _f("MSTOP_HOLD_STRICT_W", 0.0)
        self.stop_hold_contact_w = _f("MSTOP_HOLD_CONTACT_W", 0.0)
        self.stop_post_drop_c = _f("MSTOP_POST_DROP_C", 0.0)
        self.stop_hold_slip_c = _f("MSTOP_HOLD_SLIP_C", 0.0)
        self.stop_hold_lat_c = _f("MSTOP_HOLD_LAT_C", 0.0)
        self.stop_hold_yaw_c = _f("MSTOP_HOLD_YAW_C", 0.0)
        self.strict_hand_surface = _f("MSTOP_STRICT_HAND_SURFACE", .12)
        self.strict_hand_force = _f("MSTOP_STRICT_HAND_FORCE", 1.0)
        self.strict_foot_force = _f("MSTOP_STRICT_FOOT_FORCE", 20.0)
        self._mstop_diag_enabled = os.environ.get("MSTOP_DIAG", "0") == "1"
        if not 0 <= self.stop_prob <= 1:
            raise ValueError("MSTOP_STOP_PROB must be in [0, 1]")
        if (self.stop_brake_time <= 0 or self.stop_resume_time <= 0
                or self.stop_track_delta <= 0):
            raise ValueError("MSTOP_BRAKE_T, MSTOP_RESUME_T, and MSTOP_TRACK_DELTA must be positive")
        if min(self.stop_hold_strict_w, self.stop_hold_contact_w,
               self.stop_post_drop_c, self.stop_hold_slip_c,
               self.stop_hold_lat_c, self.stop_hold_yaw_c) < 0:
            raise ValueError("MSTOP HOLD quality weights must be non-negative")
        self._stop_resets = 0

        out = os.environ.get("MA_METRICS")
        if out and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            root, ext = os.path.splitext(out)
            os.environ["MA_METRICS"] = (
                f"{root}.rank{int(os.environ.get('RANK', '0'))}{ext or '.npy'}")

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self.stop_carrywith and self.stop_floor_start:
            raise ValueError("MSTOP_CARRYWITH=1 and MS_FLOOR_START=1 are mutually exclusive")
        if self.stop_carrywith:
            probs = torch.zeros_like(self._carry_skill_init_prob)
            probs[self._carry_skill.index("carryWith")] = 1.0
            self._carry_skill_init_prob = probs
        elif self.stop_floor_start:
            probs = torch.zeros_like(self._carry_skill_init_prob)
            probs[self._carry_skill.index("loco_carry")] = 1.0
            self._carry_skill_init_prob = probs
        self.stop_brake_steps = max(1, round(self.stop_brake_time / self.dt))
        self.stop_resume_steps = max(1, round(self.stop_resume_time / self.dt))
        if self.num_agents != 2:
            raise ValueError("HumanoidMAStopCarry requires numAgents=2")
        print(f"[mastop-carry] stage={self.stop_stage} mix={self.stop_mix} "
              f"stopProb={self.stop_prob:g} "
              f"aLayout={int(self.stop_a_layout)} floor={int(self.stop_floor_start)}",
              flush=True)

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        E, R = self.num_envs, self._rows
        self._stop_scen = torch.zeros(E, device=self.device, dtype=torch.long)
        self._stop_enabled = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._yield = torch.zeros(E, device=self.device, dtype=torch.long)
        self._hold = torch.ones(E, device=self.device, dtype=torch.long)
        self._latched = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._stop_done = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._stop_mask = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._resume_mask = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._stop_scale = torch.ones(R, device=self.device)
        self._brake_start = torch.full((E,), -1, device=self.device, dtype=torch.long)
        self._stop_until = torch.zeros(E, device=self.device, dtype=torch.long)
        self._release = torch.full((E,), -1, device=self.device, dtype=torch.long)
        self._resuming = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._resume_start = torch.full((E,), -1, device=self.device, dtype=torch.long)
        self._was_stopped = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._resumed = torch.zeros(E, device=self.device, dtype=torch.bool)
        self._resume_dt = torch.full((E,), -1.0, device=self.device)
        self._x_arc = torch.zeros(R, device=self.device)
        self._pick_arc0 = torch.zeros(R, device=self.device)
        self._box_vema = torch.zeros(R, device=self.device)
        self._stop_prev_box = torch.zeros(R, 2, device=self.device)
        self._met_stop_n = torch.zeros(R, device=self.device, dtype=torch.long)
        self._met_root_v = torch.zeros(R, device=self.device)
        self._met_box_v = torch.zeros(R, device=self.device)
        self._met_held = torch.zeros(R, device=self.device, dtype=torch.long)
        self._met_drop = torch.zeros(R, device=self.device, dtype=torch.long)
        self._met_hold_n = torch.zeros(R, device=self.device, dtype=torch.long)
        self._met_brake_n = torch.zeros(R, device=self.device, dtype=torch.long)
        self._met_resume_n = torch.zeros(R, device=self.device, dtype=torch.long)
        # Visual-aligned diagnostics; legacy pickup only latches a brief mean-hand proximity.
        self._strict_stop_held = torch.zeros(R, device=self.device, dtype=torch.long)
        self._strict_stop_drop = torch.zeros(R, device=self.device, dtype=torch.long)
        self._strict_resume_held = torch.zeros(R, device=self.device, dtype=torch.long)
        self._strict_resume_drop = torch.zeros(R, device=self.device, dtype=torch.long)
        self._strict_terminal = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._strict_ever = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._strict_broken = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._contact_stop_held = torch.zeros(R, device=self.device, dtype=torch.long)
        self._contact_resume_held = torch.zeros(R, device=self.device, dtype=torch.long)
        self._contact_terminal = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._strict_regrasp = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._strict_post_stop_drop = torch.zeros(R, device=self.device, dtype=torch.bool)
        self._strict_terminal_surface = torch.full((R, 2), 99.0, device=self.device)
        self._strict_terminal_lift = torch.zeros(R, device=self.device)
        self._hold_foot_slip = torch.zeros(R, device=self.device)
        self._hold_foot_contact_n = torch.zeros(R, device=self.device, dtype=torch.long)
        self._hold_root_lat_v = torch.zeros(R, device=self.device)
        self._hold_root_yaw_rate = torch.zeros(R, device=self.device)
        self._met_col = torch.zeros(E, device=self.device, dtype=torch.long)
        self._met_clear = torch.full((E,), 99.0, device=self.device)
        self._mstop_diag_stats = torch.zeros(4, 11, device=self.device)

    def _sample(self, env_ids):
        if len(env_ids) == 0:
            return
        self._stop_resets += 1
        g = torch.Generator(device="cpu")
        g.manual_seed(self.stop_seed + self._stop_resets)
        p = torch.tensor(self.stop_mix)
        self._stop_scen[env_ids] = torch.multinomial(
            p, len(env_ids), replacement=True, generator=g).to(self.device)
        enabled = torch.rand(len(env_ids), generator=g) < self.stop_prob
        self._stop_enabled[env_ids] = (
            (self._stop_scen[env_ids] != A) & enabled.to(self.device))
        self._yield[env_ids] = torch.randint(
            2, (len(env_ids),), generator=g).to(self.device)
        lo = max(1, round(self.stop_hold_min / self.dt))
        hi = max(lo + 1, round(self.stop_hold_max / self.dt) + 1)
        self._hold[env_ids] = torch.randint(
            lo, hi, (len(env_ids),), generator=g).to(self.device)

    def apply_layout(self, env_ids):
        self._sample(env_ids)
        custom = self._stop_scen[env_ids] != A
        if self.stop_a_layout:
            custom &= self._stop_scen[env_ids] != B
        env_ids = env_ids[custom]
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids)
        env = torch.div(rows, 2, rounding_mode="floor")
        agent = rows % 2
        half = torch.full((len(rows),), self.stop_len / 2, device=self.device)
        off = (agent.float() - .5) * self.stop_solo_gap
        solo_start = torch.stack([-half, off], -1)
        solo_tar = torch.stack([half, off], -1)
        cross_start = torch.stack([
            torch.where(agent == 0, -half, torch.zeros_like(half)),
            torch.where(agent == 1, -half, torch.zeros_like(half))], -1)
        solo = self._stop_scen[env] == B
        start = torch.where(solo[:, None], solo_start, cross_start)
        tar = torch.where(solo[:, None], solo_tar, -cross_start)

        origin = self.agent_axis(
            self._initial_humanoid_root_states)[:, :, :2].mean(1)[env]
        humanoid = self.agent_axis(self._humanoid_root_states)
        box = self.agent_axis(self._box_states)
        xy = origin + start
        shift = xy - humanoid[env, agent, :2]
        humanoid[env, agent, :2] = xy
        box[env, agent, :2] += shift
        self._box_tar_pos[rows, :2] = origin + tar
        if self._carry_reset_random_height:
            self.agent_axis(self._platform_states)[env, agent, :2] = box[env, agent, :2]
            self.agent_axis(
                self._tar_platform_states)[env, agent, :2] = self._box_tar_pos[rows, :2]

    def _reset_steer(self, rows):
        super()._reset_steer(rows)
        if len(rows) == 0:
            return
        env = torch.div(rows, 2, rounding_mode="floor")
        fixed_mask = self._stop_scen[env] != A
        if self.stop_a_layout:
            fixed_mask &= self._stop_scen[env] != B
        fixed = rows[fixed_mask]
        if len(fixed):
            h = self.humanoid_rows(self._humanoid_root_states)
            b = self.humanoid_rows(self._box_states)
            path, s_box, n_end = sp.gen_full_v2(
                h[fixed, :2], b[fixed, :2], self._box_tar_pos[fixed, :2],
                self.stop_seed + self._stop_resets + int(fixed[0]),
                0.0, 0.0, 120.0, p_two=.1, skew=.8,
                spread=(.85, 1.8), lat_frac=.25, with_end=True)
            self._gt_path[fixed] = path
            self._s_end[fixed] = (n_end.float() - 1) * sp.DS
            self._arc_root[fixed] = 0
            self._arc_box[fixed] = s_box

        origin = self.agent_axis(
            self._initial_humanoid_root_states)[:, :, :2].mean(1)[env]
        nearest = torch.norm(
            self._gt_path[rows] - origin[:, None], dim=-1).argmin(1)
        self._x_arc[rows] = nearest.float() * sp.DS
        self._pick_arc0[rows] = self._arc_box[rows]
        self._stop_prev_box[rows] = self.humanoid_rows(self._box_states)[rows, :2]
        self._box_vema[rows] = 0
        touched = env.unique()
        self._latched[touched] = False
        self._stop_done[touched] = False
        self._stop_mask[self.agent_rows(touched)] = False
        self._resume_mask[self.agent_rows(touched)] = False
        self._stop_scale[self.agent_rows(touched)] = 1
        self._brake_start[touched] = -1
        self._stop_until[touched] = 0
        self._release[touched] = -1
        self._resuming[touched] = False
        self._resume_start[touched] = -1
        self._was_stopped[touched] = False
        self._resumed[touched] = False
        self._resume_dt[touched] = -1

    def _update_stop(self):
        if self.stop_stage_i == 0:
            return
        env = torch.arange(self.num_envs, device=self.device)
        box = self.humanoid_rows(self._box_states)[:, :2]
        speed = torch.norm(box - self._stop_prev_box, dim=-1) / self.dt
        self._box_vema = (1 - self.stop_alpha) * self._box_vema + self.stop_alpha * speed
        self._stop_prev_box[:] = box
        yr = env * 2 + self._yield
        other = env * 2 + 1 - self._yield
        progress = self.progress_buf

        other_passed = ((self._arc_root[other] >= self._x_arc[other] + self.stop_clear)
                        & (self._arc_box[other] >= self._x_arc[other] + self.stop_clear)
                        & (self._clearance() >= 0))
        release = self._latched & (progress >= self._stop_until) & (
            (self._stop_scen == B) | ((self._stop_scen >= C) & other_passed))
        if release.any():
            self._latched[release] = False
            self._stop_done[release] = True
            self._release[release] = progress[release]
            self._resuming[release] = True
            self._resume_start[release] = progress[release]

        picked = self._ep_picked.view(self.num_envs, 2)
        if self.stop_carrywith:
            picked = self._speeds_held()[2].view(self.num_envs, 2)
        ready = (self._stop_enabled & ~self._latched
                 & ~self._stop_done & ~self._resuming)
        solo = ready & (self._stop_scen == B)
        solo &= picked[env, self._yield]
        solo &= self._arc_box[yr] >= self._pick_arc0[yr] + self.stop_after_pick
        fixed = ready & (self._stop_scen == C) & picked.all(1)
        fixed &= self._arc_box[yr] >= self._x_arc[yr] - self.stop_brake
        fixed &= self._arc_box[other] < self._x_arc[other] + self.stop_clear

        pool = ready & (self._stop_scen == D) & picked.all(1)
        rows = env[:, None] * 2 + torch.arange(2, device=self.device)[None]
        remain = (self._x_arc[rows] - self._arc_box[rows]).clamp(min=0)
        eta = remain / self._box_vema[rows].clamp(min=.25)
        pred_clear, _ = self._predicted_clearance()
        dynamic = pool & (pred_clear < 0)
        if dynamic.any():
            later = eta.argmax(1)
            later = torch.where(
                (eta[:, 0] - eta[:, 1]).abs() < .05, self._yield, later)
            self._yield[dynamic] = later[dynamic]
            yr = env * 2 + self._yield

        activate = solo | fixed | dynamic
        self._latched[activate] = True
        self._brake_start[activate] = progress[activate]
        self._stop_until[activate] = (
            progress[activate] + self.stop_brake_steps + self._hold[activate])

        self._stop_mask[:] = False
        self._resume_mask[:] = False
        self._stop_scale[:] = 1
        active = env[self._latched]
        active_rows = active * 2 + self._yield[active]
        self._stop_mask[active_rows] = True
        elapsed = (progress[active] - self._brake_start[active]).float()
        phase = (elapsed / self.stop_brake_steps).clamp(0, 1)
        self._stop_scale[active_rows] = .5 * (1 + torch.cos(torch.pi * phase))
        hold_env = active[phase >= 1]
        if len(hold_env):
            hold_rows = hold_env * 2 + self._yield[hold_env]
            root_speed = torch.norm(
                self.humanoid_rows(self._humanoid_root_states)[:, 7:9], dim=-1)
            stopped = ((root_speed[hold_rows] < self.stop_speed)
                       & (self._box_vema[hold_rows] < self.stop_speed))
            self._was_stopped[hold_env] |= stopped

        resume_env = env[self._resuming]
        resume_rows = resume_env * 2 + self._yield[resume_env]
        resume_elapsed = (progress[resume_env] - self._resume_start[resume_env]).float()
        resume_phase = (resume_elapsed / self.stop_resume_steps).clamp(0, 1)
        self._resume_mask[resume_rows] = True
        self._stop_scale[resume_rows] = .5 * (1 - torch.cos(torch.pi * resume_phase))
        finished = resume_elapsed >= self.stop_resume_steps
        if finished.any():
            done_env = resume_env[finished]
            done_rows = done_env * 2 + self._yield[done_env]
            self._resuming[done_env] = False
            self._resume_mask[done_rows] = False
            self._stop_scale[done_rows] = 1

        pending = (self._release >= 0) & ~self._resumed
        if pending.any():
            root_v = torch.norm(
                self.humanoid_rows(self._humanoid_root_states)[:, 7:9], dim=-1)
            command_ready = (
                self._stop_scale[yr]
                >= self.stop_resume_v / (self.steer_m_nom / 1.6))
            resumed = pending & self._was_stopped & command_ready & (
                (root_v[yr] > self.stop_resume_v)
                & (self._box_vema[yr] > self.stop_resume_v))
            self._resumed[resumed] = True
            self._resume_dt[resumed] = (
                progress[resumed] - self._release[resumed]).float() * self.dt

    def _compute_observations(self, env_ids=None):
        if env_ids is None and hasattr(self, "_stop_mask"):
            self._update_stop()
        return super()._compute_observations(env_ids)
    def _compute_task_obs(self, env_ids=None):
        obs = super()._compute_task_obs(env_ids)
        if not hasattr(self, "_stop_mask"):
            return obs
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        active = self._stop_mask[rows] | self._resume_mask[rows]
        scale = self._stop_scale[rows]
        gate = torch.where(active, 1.0 - scale, torch.zeros_like(scale))
        brake = self._stop_mask[rows] & (scale > 1e-4)
        hold = self._stop_mask[rows] & ~brake
        resume = self._resume_mask[rows]
        held = self._speeds_held()[2][rows]
        command = torch.stack((gate, scale, brake.float(), hold.float(),
                               resume.float(), held.float()), dim=-1)
        obs = obs.clone()
        obs[:, :TEAMMATE_DIM] = 0
        obs[:, :6] = command
        return obs


    def _m_at(self, arc, rows=None):
        command = super()._m_at(arc, rows)
        rows = torch.arange(len(arc), device=self.device) if rows is None else rows
        return command * self._stop_scale[rows]

    def _clearance(self):
        h = self.agent_axis(self._humanoid_root_states)[:, :, :2]
        b = self.agent_axis(self._box_states)[:, :, :2]
        size = self._box_lib._box_size.view(self.num_envs, 2, 3)
        radius = torch.norm(size[:, :, :2], dim=-1) * .5
        return torch.stack([
            torch.norm(h[:, 0] - h[:, 1], dim=-1) - .7,
            torch.norm(h[:, 0] - b[:, 1], dim=-1) - .35 - radius[:, 1],
            torch.norm(b[:, 0] - h[:, 1], dim=-1) - radius[:, 0] - .35,
            torch.norm(b[:, 0] - b[:, 1], dim=-1) - radius[:, 0] - radius[:, 1],
        ], -1).min(-1).values

    def _predicted_clearance(self):
        h = self.agent_axis(self._humanoid_root_states)
        b = self.agent_axis(self._box_states)
        size = self._box_lib._box_size.view(self.num_envs, 2, 3)
        radius = torch.norm(size[:, :, :2], dim=-1) * .5
        t = torch.linspace(0, self.stop_horizon, 9, device=self.device)
        hp = h[:, :, None, :2] + h[:, :, None, 7:9] * t[None, None, :, None]
        bp = b[:, :, None, :2] + b[:, :, None, 7:9] * t[None, None, :, None]
        clear = torch.stack([
            torch.norm(hp[:, 0] - hp[:, 1], dim=-1) - .7,
            torch.norm(hp[:, 0] - bp[:, 1], dim=-1) - .35 - radius[:, 1, None],
            torch.norm(bp[:, 0] - hp[:, 1], dim=-1) - radius[:, 0, None] - .35,
            torch.norm(bp[:, 0] - bp[:, 1], dim=-1)
            - radius[:, 0, None] - radius[:, 1, None],
        ], -1).min(-1).values
        value, index = clear.min(-1)
        return value, t[index]

    def _speeds_held(self):
        h = self.humanoid_rows(self._humanoid_root_states)
        b = self.humanoid_rows(self._box_states)
        root_v = torch.norm((h[:, :2] - self._prev_root_pos[:, :2]) / self.dt, dim=-1)
        box_v = torch.norm((b[:, :2] - self._prev_box_pos[:, :2]) / self.dt, dim=-1)
        rb = self.humanoid_rows(self._rigid_body_pos)
        hands = rb[:, self._key_body_ids[[0, 1]]].mean(1)
        close = torch.norm(hands - b[:, :3], dim=-1) < self.pick_hand_dist
        lifted = (b[:, 2] - self._pick_z0) >= self.pick_lift
        held = close if self.stop_carrywith else close & lifted
        return root_v, box_v, held

    def _strict_grasp(self):
        """Two-hand OBB proximity plus hand-force grasp diagnostic."""
        b = self.humanoid_rows(self._box_states)
        rb = self.humanoid_rows(self._rigid_body_pos)
        hands = rb[:, self._key_body_ids[[0, 1]]]
        rot = b[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        local = quat_rotate_inverse(
            rot, (hands - b[:, None, :3]).reshape(-1, 3)).view(-1, 2, 3)
        half = self._box_lib._box_size.view(-1, 3)[:, None, :] * .5
        surface = torch.norm(torch.clamp(local.abs() - half, min=0), dim=-1)
        geometry = (surface <= self.strict_hand_surface).all(dim=-1)

        forces = self.humanoid_rows(self._contact_forces)
        hand_force = torch.norm(forces[:, self._key_body_ids[[0, 1]]], dim=-1)
        contact = (hand_force >= self.strict_hand_force).all(dim=-1)
        lift = b[:, 2] - self._pick_z0
        lifted = torch.ones_like(geometry) if self.stop_carrywith else lift >= self.pick_lift
        strict = geometry & lifted
        return strict, strict & contact, surface, lift

    def _hold_motion_quality(self):
        """Contact-foot slip plus path-frame lateral/yaw motion per row."""
        vel = self.humanoid_rows(self._rigid_body_vel)
        force = self.humanoid_rows(self._contact_forces)
        feet = self._key_body_ids[[2, 3]]
        foot_contact = force[:, feet, 2].abs() >= self.strict_foot_force
        foot_speed = torch.norm(vel[:, feet, :2], dim=-1)
        slip_sum = torch.where(foot_contact, foot_speed, torch.zeros_like(foot_speed)).sum(1)
        contact_n = foot_contact.sum(1)

        h = self.humanoid_rows(self._humanoid_root_states)
        aim = self._aim_pt(self._arc_root, torch.full_like(self._arc_root, .5))
        forward = torch.nn.functional.normalize(aim - h[:, :2], dim=-1)
        lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        root_lat_v = (h[:, 7:9] * lateral).sum(-1).abs()
        yaw_rate = h[:, 12].abs()
        return slip_sum, contact_n, root_lat_v, yaw_rate

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        base_carry_reward = self.rew_buf.clone()
        active = self._stop_mask | self._resume_mask
        stop_bonus = torch.zeros_like(self.rew_buf)
        if active.any():
            root_v, box_v, held = self._speeds_held()
            hold = self._stop_mask & (self._stop_scale <= 1e-4)
            target_v = self.steer_m_nom * self._stop_scale / 1.6
            root_err = torch.abs(root_v - target_v)
            box_err = torch.abs(box_v - target_v)
            delta = self.stop_track_delta
            root_loss = torch.where(root_err < delta,
                                    .5 * root_err.square() / delta, root_err - .5 * delta)
            box_loss = torch.where(box_err < delta,
                                   .5 * box_err.square() / delta, box_err - .5 * delta)
            transition_bonus = (
                self.stop_hold_w * held.float()
                - self.stop_drop_c * (~held).float()
                - self.stop_track_w * .5 * (root_loss + box_loss))
            hold_bonus = (
                self.stop_root_w * torch.exp(-4 * root_v.square())
                + self.stop_box_w * torch.exp(-4 * box_v.square())
                + self.stop_hold_w * held.float()
                - self.stop_creep_c * (root_v + box_v)
                - self.stop_drop_c * (~held).float())
            if (self.stop_hold_strict_w or self.stop_hold_slip_c
                    or self.stop_hold_lat_c or self.stop_hold_yaw_c):
                _, _, hand_surface, _ = self._strict_grasp()
                worst_hand = hand_surface.max(dim=-1).values
                both_hand = torch.exp(-.5 * (worst_hand / self.strict_hand_surface).square())
                slip_sum, contact_n, lateral_v, yaw_rate = self._hold_motion_quality()
                slip = slip_sum / contact_n.clamp(min=1)
                hold_bonus = (
                    hold_bonus
                    + self.stop_hold_strict_w * both_hand
                    - self.stop_hold_slip_c * slip
                    - self.stop_hold_lat_c * lateral_v
                    - self.stop_hold_yaw_c * yaw_rate)
            bonus = torch.where(hold, hold_bonus, transition_bonus)
            if self.stop_hold_contact_w or self.stop_post_drop_c:
                strict, contact_grasp, _, _ = self._strict_grasp()
                bonus = (bonus
                         + self.stop_hold_contact_w * contact_grasp.float()
                         - self.stop_post_drop_c * (~strict).float())
            stop_bonus = torch.where(active, bonus, torch.zeros_like(bonus))
            self.rew_buf += stop_bonus
        if self.stop_stage_i >= 2:
            conflict = self._stop_scen >= C
            penalty = self.stop_collision_c * (
                conflict & (self._clearance() < 0)).float()
            self.rew_buf -= penalty.repeat_interleave(2)

        if self._mstop_diag_enabled:
            root_v, box_v, held = self._speeds_held()
            target_v = self.steer_m_nom * self._stop_scale / 1.6
            brake = self._stop_mask & (self._stop_scale > 1e-4)
            hold = self._stop_mask & ~brake
            phase = torch.zeros(self._rows, device=self.device, dtype=torch.long)
            phase[brake] = 1
            phase[hold] = 2
            phase[self._resume_mask] = 3
            values = torch.stack((
                torch.ones_like(root_v), root_v, box_v, target_v,
                (root_v - target_v).abs(), (box_v - target_v).abs(),
                held.float(), (~held).float(), base_carry_reward, stop_bonus,
                self.rew_buf), dim=-1)
            for i in range(4):
                self._mstop_diag_stats[i] += values[phase == i].sum(0)

    def consume_mstop_diag(self):
        stats = self._mstop_diag_stats.clone()
        self._mstop_diag_stats.zero_()
        return stats

    def update_metrics(self):
        super().update_metrics()
        root_v, box_v, held = self._speeds_held()
        strict, contact_grasp, surface, lift = self._strict_grasp()
        stop = self._stop_mask
        hold = stop & (self._stop_scale <= 1e-4)
        brake = stop & ~hold
        self._met_stop_n += stop.long()
        self._met_hold_n += hold.long()
        self._met_brake_n += brake.long()
        self._met_resume_n += self._resume_mask.long()
        self._met_root_v += torch.where(hold, root_v, torch.zeros_like(root_v))
        self._met_box_v += torch.where(hold, box_v, torch.zeros_like(box_v))
        self._met_held += (stop & held).long()
        self._strict_stop_held += (stop & strict).long()
        self._strict_stop_drop += (stop & ~strict).long()
        self._contact_stop_held += (stop & contact_grasp).long()
        self._contact_resume_held += (self._resume_mask & contact_grasp).long()
        self._contact_terminal[:] = contact_grasp
        self._strict_resume_held += (self._resume_mask & strict).long()
        self._strict_resume_drop += (self._resume_mask & ~strict).long()

        # Start break accounting only after a strict grasp has existed once.
        was_ever = self._strict_ever.clone()
        was_broken = self._strict_broken.clone()
        self._strict_ever |= self._ep_picked & strict
        self._strict_broken |= was_ever & ~strict
        self._strict_regrasp |= was_broken & strict
        env = torch.arange(self.num_envs, device=self.device)
        yield_rows = env * 2 + self._yield
        after_stop = torch.zeros(self._rows, device=self.device, dtype=torch.bool)
        after_stop[yield_rows] = self._stop_done | self._latched | self._resuming
        self._strict_post_stop_drop |= after_stop & ~strict
        self._strict_terminal[:] = strict
        self._strict_terminal_surface[:] = surface
        self._strict_terminal_lift[:] = lift

        slip_sum, contact_n, root_lat_v, yaw_rate = self._hold_motion_quality()
        self._hold_foot_slip += torch.where(hold, slip_sum, torch.zeros_like(slip_sum))
        self._hold_foot_contact_n += torch.where(
            hold, contact_n, torch.zeros_like(contact_n))
        self._hold_root_lat_v += torch.where(hold, root_lat_v, torch.zeros_like(root_lat_v))
        self._hold_root_yaw_rate += torch.where(hold, yaw_rate, torch.zeros_like(yaw_rate))
        self._met_drop += (stop & ~held).long()
        clear = self._clearance()
        self._met_clear = torch.minimum(self._met_clear, clear)
        self._met_col += (clear < 0).long()

    def _metric_extra_cols(self, rows):
        cols = super()._metric_extra_cols(rows)
        env = torch.div(rows, 2, rounding_mode="floor")
        cols += [
            self._stop_scen[env].float(),
            ((rows % 2) == self._yield[env]).float(),
            self._met_stop_n[rows].float(),
            self._met_root_v[rows],
            self._met_box_v[rows],
            self._met_held[rows].float(),
            self._met_drop[rows].float(),
            self._resumed[env].float(),
            self._resume_dt[env],
            self._met_col[env].float(),
            self._met_clear[env],
            self._met_hold_n[rows].float(),
            self._met_brake_n[rows].float(),
            self._met_resume_n[rows].float(),
            self._strict_stop_held[rows].float(),
            self._strict_stop_drop[rows].float(),
            self._strict_resume_held[rows].float(),
            self._strict_resume_drop[rows].float(),
            self._strict_terminal[rows].float(),
            self._strict_ever[rows].float(),
            self._strict_broken[rows].float(),
            self._strict_regrasp[rows].float(),
            self._strict_post_stop_drop[rows].float(),
            self._strict_terminal_surface[rows, 0],
            self._strict_terminal_surface[rows, 1],
            self._strict_terminal_lift[rows],
            self._hold_foot_slip[rows],
            self._contact_stop_held[rows].float(),
            self._contact_resume_held[rows].float(),
            self._contact_terminal[rows].float(),
            self._hold_foot_contact_n[rows].float(),
            self._hold_root_lat_v[rows],
            self._hold_root_yaw_rate[rows],
        ]
        return cols

    def _metric_reset_extra(self, rows):
        super()._metric_reset_extra(rows)
        self._met_stop_n[rows] = 0
        self._met_root_v[rows] = 0
        self._met_box_v[rows] = 0
        self._met_held[rows] = 0
        self._met_drop[rows] = 0
        self._met_hold_n[rows] = 0
        self._met_brake_n[rows] = 0
        self._met_resume_n[rows] = 0
        self._strict_stop_held[rows] = 0
        self._contact_stop_held[rows] = 0
        self._contact_resume_held[rows] = 0
        self._contact_terminal[rows] = False
        self._strict_stop_drop[rows] = 0
        self._strict_resume_held[rows] = 0
        self._strict_resume_drop[rows] = 0
        self._strict_terminal[rows] = False
        self._strict_ever[rows] = False
        self._strict_broken[rows] = False
        self._strict_regrasp[rows] = False
        self._strict_post_stop_drop[rows] = False
        self._strict_terminal_surface[rows] = 99.0
        self._strict_terminal_lift[rows] = 0.0
        self._hold_foot_slip[rows] = 0.0
        self._hold_foot_contact_n[rows] = 0
        self._hold_root_lat_v[rows] = 0.0
        self._hold_root_yaw_rate[rows] = 0.0
        env = torch.div(rows, 2, rounding_mode="floor").unique()
        self._met_col[env] = 0
        self._met_clear[env] = 99
