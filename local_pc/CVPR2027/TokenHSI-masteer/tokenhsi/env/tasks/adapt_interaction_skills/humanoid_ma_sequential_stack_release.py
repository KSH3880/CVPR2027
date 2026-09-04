"""Shared-policy sequential stacking with gated release and retreat rewards.

The centralized controller assigns base/top roles, stages the top carrier, and
retargets existing steering commands.  The actor observation/model layout is
identical to :class:`HumanoidMASteerCarry`; no role or phase token is added.

Both agents use the exact parent carry reward until the base box is placed.
Only the current base carrier then receives the release/clear replacement
reward.  Box sizes are sampled from the native BoxLib pool, subject only to the
top footprint fitting on the base, and roles are balanced across agent indices.
"""

import math
import os

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp


def _f(name, default):
    value = os.environ.get(name)
    return float(default if value is None or value.strip() == "" else value)


def _i(name, default):
    value = os.environ.get(name)
    return int(default if value is None or value.strip() == "" else value)


class HumanoidMASequentialStackRelease(HumanoidMASteerCarry):
    """Two-agent carry -> base release -> body clear -> top placement task."""

    CARRY = 0
    RELEASE = 1
    CLEAR = 2
    STACK = 3
    SUCCESS = 4
    FAILED = 5

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if int(cfg["env"].get("numAgents", 1)) != 2:
            raise ValueError("HumanoidMASequentialStackRelease requires env.numAgents=2")

        self._ss_seed = _i("STACK_SEED", int(cfg.get("seed", 0)) + 19020)
        self._ss_size_margin = _f("STACK_SIZE_MARGIN", 0.05)
        self._ss_goal_x = _f("STACK_GOAL_X", 4.0)
        self._ss_goal_y = _f("STACK_GOAL_Y", 0.0)
        self._ss_stage_dist = _f("STACK_STAGE_DIST", 2.0)
        self._ss_stage_z = _f("STACK_STAGE_Z", 0.90)
        self._ss_stage_tol = _f("STACK_STAGE_TOL", 0.45)
        self._ss_body_clear = _f("STACK_BODY_CLEAR", 1.0)
        self._ss_retreat_dist = _f("STACK_RETREAT_DIST", 1.5)
        # Keep the old implicit value as the class default so ms20 sidecars
        # remain replayable; the ms21 wrapper explicitly selects 0.5.
        self._ss_retreat_scale = _f("STACK_RETREAT_SCALE", 1.0)

        self._ss_xy_tol = _f("STACK_XY_TOL", 0.10)
        self._ss_z_tol = _f("STACK_Z_TOL", 0.06)
        self._ss_entry_lin = _f("STACK_ENTRY_LIN", 0.25)
        self._ss_entry_ang = _f("STACK_ENTRY_ANG", 1.0)
        self._ss_entry_steps = _i("STACK_ENTRY_STEPS", 5)
        self._ss_hand_clear = _f("STACK_HAND_CLEAR", 0.15)
        self._ss_hand_steps = _i("STACK_HAND_STEPS", 5)
        self._ss_stable_lin = _f("STACK_STABLE_LIN", 0.08)
        self._ss_stable_ang = _f("STACK_STABLE_ANG", 0.20)
        self._ss_top_steps = _i("STACK_TOP_STEPS", 20)
        self._ss_drop_xy = _f("STACK_DROP_XY", 0.25)
        self._ss_transition_bonus = _f("STACK_TRANSITION_BONUS", 3.0)
        self._ss_clear_bonus = _f("STACK_CLEAR_BONUS", 1.5)
        self._ss_clear_progress_w = _f("STACK_CLEAR_PROGRESS_W", 0.7)
        self._ss_clear_speed_w = _f("STACK_CLEAR_SPEED_W", 0.3)
        self._ss_clear_vel_k = _f("STACK_CLEAR_VEL_K", 4.0)
        self._ss_clear_lat_k = _f("STACK_CLEAR_LAT_K", 1.0)
        self._ss_clear_min_speed = _f("STACK_CLEAR_MIN_SPEED", 0.05)
        self._ss_clear_hard_gate = bool(_i("STACK_CLEAR_HARD_GATE", 0))
        if self._ss_retreat_scale <= 0.0 or self._ss_retreat_scale > 1.0:
            raise ValueError("STACK_RETREAT_SCALE must be in (0, 1]")
        if self._ss_clear_progress_w < 0.0 or self._ss_clear_speed_w < 0.0:
            raise ValueError("STACK_CLEAR_PROGRESS_W/STACK_CLEAR_SPEED_W must be non-negative")
        clear_w = self._ss_clear_progress_w + self._ss_clear_speed_w
        if clear_w <= 0.0:
            raise ValueError("at least one CLEAR motion reward weight must be positive")
        self._ss_clear_progress_w /= clear_w
        self._ss_clear_speed_w /= clear_w

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    def register_task_carry_post_init(self, cfg):
        """Every rollout starts before pickup so the dependency is well-defined."""
        super().register_task_carry_post_init(cfg)
        self._carry_skill_init_prob[:] = 0.0
        self._carry_skill_init_prob[0] = 1.0

    # ------------------------------------------------------------ assets/roles

    def _load_box_asset(self, box_sizes):
        """Sample valid native-size pairs and balance roles across agent IDs."""
        if box_sizes.shape[0] % 2:
            raise ValueError("stack release requires an even number of box rows")

        lib = self._box_lib
        if lib.mode == "test":
            pool = np.asarray(lib._build_test_sizes, dtype=np.float32)
        elif lib._build_random_mode_equal_proportion:
            scales = np.arange(
                lib._build_x_scale_range[0],
                lib._build_x_scale_range[1] + 0.5 * lib._build_scale_sample_interval,
                lib._build_scale_sample_interval,
                dtype=np.float32,
            )
            pool = np.asarray(lib._build_base_size, dtype=np.float32)[None, :] * scales[:, None]
        else:
            # Fall back to the already sampled native pool for a future
            # non-equal-proportion configuration.
            pool = np.unique(
                np.round(box_sizes.detach().cpu().numpy(), decimals=6), axis=0
            )
        valid = []
        for ib, base in enumerate(pool):
            for it, top in enumerate(pool):
                if np.all(base[:2] >= top[:2] + self._ss_size_margin):
                    valid.append((ib, it))
        if not valid:
            raise ValueError(
                "native box-size pool has no stackable pair; reduce STACK_SIZE_MARGIN"
            )

        rng = np.random.default_rng(self._ss_seed)
        envs = box_sizes.shape[0] // 2
        pair_ids = rng.integers(0, len(valid), size=envs)
        base_agent = rng.integers(0, 2, size=envs, dtype=np.int64)
        # Exact balance removes an avoidable agent-index correlation.
        base_agent[: envs // 2] = 0
        base_agent[envs // 2 :] = 1
        rng.shuffle(base_agent)

        sampled = np.empty((envs, 2, 3), dtype=np.float32)
        for e, pair_id in enumerate(pair_ids):
            ib, it = valid[int(pair_id)]
            b, t = int(base_agent[e]), 1 - int(base_agent[e])
            sampled[e, b] = pool[ib]
            sampled[e, t] = pool[it]
        box_sizes[:] = torch.as_tensor(sampled.reshape(-1, 3), device=box_sizes.device)
        self._box_lib._build_box_bps()
        self._ss_base_agent_np = base_agent
        return super()._load_box_asset(box_sizes)

    # --------------------------------------------------------------- lifecycle

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        if len(self._ss_base_agent_np) != num_envs:
            raise RuntimeError("base-role sampling does not match num_envs")
        self._ss_base_agent = torch.as_tensor(
            self._ss_base_agent_np, dtype=torch.long, device=self.device
        )
        self._ss_phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_entry_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_hand_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_top_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_staged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_bonus_pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_clear_bonus_pending = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_base_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_stage_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_top_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_latched_base = torch.zeros((num_envs, 3), device=self.device)
        print(
            "[stack-release] shared policy; balanced random base/top roles; "
            f"transition_bonus={self._ss_transition_bonus:.2f} "
            f"clear_bonus={self._ss_clear_bonus:.2f} "
            f"retreat_scale={self._ss_retreat_scale:.2f}",
            flush=True,
        )
        return

    def _role_rows(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        base = env_ids * self.num_agents + self._ss_base_agent[env_ids]
        top = env_ids * self.num_agents + 1 - self._ss_base_agent[env_ids]
        return base, top

    def _sync_stack_actors(self, env_ids):
        rows = self.agent_rows(env_ids)
        actor_ids = [self._box_actor_ids[rows]]
        if self._carry_reset_random_height:
            actor_ids.extend([
                self._platform_actor_ids[rows],
                self._tar_platform_actor_ids[rows],
            ])
        actor_ids = torch.cat(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return

        boxes = self.agent_axis(self._box_states)
        sizes = self._box_lib._box_size.view(self.num_envs, self.num_agents, 3)
        base_rows, top_rows = self._role_rows(env_ids)
        base_a = self._ss_base_agent[env_ids]
        top_a = 1 - base_a

        # This task uses the two dynamic boxes as the only supports.
        if self._carry_reset_random_height:
            for states in (self._platform_states, self._tar_platform_states):
                shaped = self.agent_axis(states)
                shaped[env_ids, :, 0:2] = 0.0
                shaped[env_ids, :, 2] = -10.0
                shaped[env_ids, :, 7:13] = 0.0

        boxes[env_ids, :, 2] = 0.5 * sizes[env_ids, :, 2]
        boxes[env_ids, :, 7:13] = 0.0

        initial = self.agent_axis(self._initial_humanoid_root_states)
        origin = initial[env_ids, :, 0:2].mean(dim=1)
        goal_xy = origin + torch.tensor(
            [self._ss_goal_x, self._ss_goal_y], device=self.device
        )
        base_size = sizes[env_ids, base_a]
        top_size = sizes[env_ids, top_a]

        self._ss_base_goal[env_ids, 0:2] = goal_xy
        self._ss_base_goal[env_ids, 2] = 0.5 * base_size[:, 2]
        self._ss_top_goal[env_ids, 0:2] = goal_xy
        self._ss_top_goal[env_ids, 2] = base_size[:, 2] + 0.5 * top_size[:, 2]

        side = torch.where(top_a == 0, -torch.ones_like(top_a), torch.ones_like(top_a)).float()
        self._ss_stage_goal[env_ids, 0] = goal_xy[:, 0]
        self._ss_stage_goal[env_ids, 1] = goal_xy[:, 1] + side * self._ss_stage_dist
        self._ss_stage_goal[env_ids, 2] = torch.maximum(
            self._ss_top_goal[env_ids, 2],
            torch.full_like(side, self._ss_stage_z),
        )

        self._box_tar_pos[base_rows] = self._ss_base_goal[env_ids]
        self._box_tar_pos[top_rows] = self._ss_stage_goal[env_ids]

        self._ss_phase[env_ids] = self.CARRY
        self._ss_entry_count[env_ids] = 0
        self._ss_hand_count[env_ids] = 0
        self._ss_top_count[env_ids] = 0
        self._ss_staged[env_ids] = False
        self._ss_bonus_pending[env_ids] = False
        self._ss_clear_bonus_pending[env_ids] = False
        self._ss_success[env_ids] = False
        self._ss_failed[env_ids] = False
        self._ss_latched_base[env_ids] = 0.0

        self._sync_stack_actors(env_ids)
        self._reset_steer(self.agent_rows(env_ids))
        self._mscale[base_rows] = 1.0
        self._mscale[top_rows] = 1.0
        self._reset_progress_ref(env_ids)
        return

    # ------------------------------------------------------------- measurements

    def _hand_surface_distance(self, rows):
        box = self.humanoid_rows(self._box_states)[rows]
        rb = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands = rb[:, self._key_body_ids[[0, 1]], :]
        rel = hands - box[:, None, 0:3]
        inv = quat_conjugate(box[:, 3:7])[:, None, :].expand(-1, 2, -1)
        local = quat_rotate(inv.reshape(-1, 4), rel.reshape(-1, 3)).view(-1, 2, 3)
        half = 0.5 * self._box_lib._box_size[rows, None, :]
        outside = torch.clamp(local.abs() - half, min=0.0)
        # Minimum over hands: both hands must clear the threshold.
        return outside.norm(dim=-1).amin(dim=-1)

    @staticmethod
    def _upright_score(quat):
        up = torch.zeros((len(quat), 3), device=quat.device)
        up[:, 2] = 1.0
        return quat_rotate(quat, up)[:, 2].clamp(0.0, 1.0)

    def _base_features(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        base_rows, top_rows = self._role_rows(env_ids)
        states = self.humanoid_rows(self._box_states)
        roots = self.humanoid_rows(self._humanoid_root_states)
        base, top = states[base_rows], states[top_rows]
        hand_dist = self._hand_surface_distance(base_rows)
        xy_err = (base[:, 0:2] - self._ss_base_goal[:, 0:2]).norm(dim=-1)
        z_err = (base[:, 2] - self._ss_base_goal[:, 2]).abs()
        support = torch.exp(-20.0 * xy_err.square() - 80.0 * z_err.square())
        support = support * self._upright_score(base[:, 3:7])
        stable = torch.exp(
            -10.0 * base[:, 7:10].square().sum(dim=-1)
            -0.5 * base[:, 10:13].square().sum(dim=-1)
        )
        root_dist = (roots[base_rows, 0:2] - base[:, 0:2]).norm(dim=-1)
        return base_rows, top_rows, base, top, hand_dist, xy_err, z_err, support, stable, root_dist

    # ---------------------------------------------------------------- rewards

    def _compute_reward(self, actions):
        # Every row receives exact ms18 carry first.  Only the base role is
        # replaced after a physically stable placement has been recognized.
        super()._compute_reward(actions)
        post = (self._ss_phase >= self.RELEASE) & (self._ss_phase <= self.SUCCESS)
        failed = self._ss_phase == self.FAILED
        if not (bool(post.any()) or bool(failed.any())):
            return

        (base_rows, _, _, _, hand_dist, _, _, support,
         stable, root_dist) = self._base_features()
        h = torch.clamp(hand_dist / self._ss_hand_clear, 0.0, 1.0)
        clear_now = hand_dist >= self._ss_hand_clear
        clear_score = torch.clamp(
            (root_dist - 0.40) / max(self._ss_body_clear - 0.40, 1e-6), 0.0, 1.0
        )

        # CLEAR uses the retreat path already present in the observation.  An
        # absolute distance reward pays the policy forever for standing still;
        # signed arc progress pays only motion in the commanded direction.
        arc = self._arc_root[base_rows]
        prev_arc = self._prev_arc[base_rows]
        arc_speed = (arc - prev_arc) / self.dt
        cmd_speed = self._m_at(arc, base_rows) / 1.6
        progress = (arc_speed / cmd_speed.clamp(min=1e-4)).clamp(-1.0, 1.0)
        speed_match = torch.exp(
            -self._ss_clear_vel_k * (cmd_speed - arc_speed).square()
        )
        speed_match = torch.where(
            arc_speed >= self._ss_clear_min_speed,
            speed_match,
            torch.zeros_like(speed_match),
        )
        path_quality = torch.exp(
            -self._ss_clear_lat_k * self._lat_root[base_rows].square()
        )
        physical_gate = clear_now.float() * support * stable * path_quality
        clear_motion = (
            self._ss_clear_progress_w * progress
            + self._ss_clear_speed_w * speed_match
        )

        # At contact this is exactly zero.  Support/stability become valuable
        # only while both hands are currently clear; the gate is not latched.
        release_r = 0.50 * h + clear_now.float() * (0.30 * support + 0.20 * stable)
        clear_r = physical_gate * clear_motion
        stack_r = clear_now.float() * (0.40 * support + 0.25 * stable) + 0.35 * clear_score
        replacement = torch.where(self._ss_phase == self.RELEASE, release_r, clear_r)
        replacement = torch.where(self._ss_phase >= self.STACK, stack_r, replacement)
        replacement = torch.where(
            self._ss_phase == self.SUCCESS, replacement + 0.5, replacement
        )
        replacement = replacement + self._ss_bonus_pending.float() * self._ss_transition_bonus
        replacement = (
            replacement
            + self._ss_clear_bonus_pending.float() * self._ss_clear_bonus
        )
        self._ss_bonus_pending[:] = False
        self._ss_clear_bonus_pending[:] = False
        self.rew_buf[base_rows[post]] = replacement[post]

        self.extras["stack_clear_arc_speed"] = arc_speed
        self.extras["stack_clear_cmd_speed"] = cmd_speed
        self.extras["stack_clear_path_quality"] = path_quality
        self.extras["stack_clear_physical_gate"] = physical_gate

        if bool(failed.any()):
            self.rew_buf[base_rows[failed]] = -1.0
        return

    # --------------------------------------------------------------- controller

    def _set_retreat_path(self, env_ids, base_rows, base):
        roots = self.humanoid_rows(self._humanoid_root_states)[base_rows, 0:2]
        away = roots - base[:, 0:2]
        norm = away.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(away)
        fallback[:, 0] = 1.0
        away = torch.where(norm > 1e-4, away / norm.clamp(min=1e-4), fallback)
        clear = base[:, 0:2] + away * self._ss_retreat_dist
        waypoint = roots + away * 0.15
        self._steer_tick += 1
        path, s_waypoint, s_end = sp.gen_full_v2(
            roots, waypoint, clear, self.steer_seed + self._steer_tick + int(env_ids[0]),
            0.0, 0.35, 120.0, p_two=0.0, skew=0.8,
            spread=(0.85, 1.8), lat_frac=0.25, with_end=True,
        )
        self._gt_path[base_rows] = path
        self._s_end[base_rows] = s_end
        self._arc_root[base_rows] = 0.0
        self._arc_box[base_rows] = s_waypoint
        self._prev_arc[base_rows] = 0.0
        self._mscale[base_rows] = self._ss_retreat_scale

    def _update_stack_controller(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        (base_rows, top_rows, base, top, hand_dist, xy_err, z_err,
         _, _, root_dist) = self._base_features()

        carry = self._ss_phase == self.CARRY
        placeable = (
            (xy_err <= self._ss_xy_tol)
            & (z_err <= self._ss_z_tol)
            & (base[:, 7:10].norm(dim=-1) <= self._ss_entry_lin)
            & (base[:, 10:13].norm(dim=-1) <= self._ss_entry_ang)
            & (self._upright_score(base[:, 3:7]) >= math.cos(math.radians(10.0)))
        )
        self._ss_entry_count = torch.where(
            carry & placeable,
            self._ss_entry_count + 1,
            torch.where(carry, torch.zeros_like(self._ss_entry_count), self._ss_entry_count),
        )
        enter_release = carry & (self._ss_entry_count >= self._ss_entry_steps)
        if bool(enter_release.any()):
            ids = torch.nonzero(enter_release, as_tuple=False).squeeze(-1)
            rows = base_rows[ids]
            self._ss_phase[ids] = self.RELEASE
            self._ss_latched_base[ids] = base[ids, 0:3]
            # Existing steering token only: zero window means stop the object.
            self._mscale[rows] = 0.0
            self._compute_observations(ids)

        # The top carrier performs ordinary carry to a high staging point and
        # waits there; it never needs to put down and re-grasp.
        stage_dist = (top[:, 0:3] - self._ss_stage_goal).norm(dim=-1)
        newly_staged = (~self._ss_staged) & (stage_dist <= self._ss_stage_tol)
        if bool(newly_staged.any()):
            ids = torch.nonzero(newly_staged, as_tuple=False).squeeze(-1)
            self._ss_staged[ids] = True
            self._mscale[top_rows[ids]] = 0.0
            self._compute_observations(ids)

        release = self._ss_phase == self.RELEASE
        clear_now = hand_dist >= self._ss_hand_clear
        self._ss_hand_count = torch.where(
            release & clear_now,
            self._ss_hand_count + 1,
            torch.where(release, torch.zeros_like(self._ss_hand_count), self._ss_hand_count),
        )
        just_released = release & (self._ss_hand_count >= self._ss_hand_steps)
        if bool(just_released.any()):
            ids = torch.nonzero(just_released, as_tuple=False).squeeze(-1)
            rows = base_rows[ids]
            self._ss_phase[ids] = self.CLEAR
            self._ss_bonus_pending[ids] = True
            self._set_retreat_path(ids, rows, base[ids])
            self._compute_observations(ids)

        clear = self._ss_phase == self.CLEAR
        base_still_supported = (
            (xy_err <= self._ss_xy_tol)
            & (z_err <= self._ss_z_tol)
            & (base[:, 7:10].norm(dim=-1) <= self._ss_stable_lin)
            & (base[:, 10:13].norm(dim=-1) <= self._ss_stable_ang)
        )
        ready_to_stack = clear & clear_now & (root_dist >= self._ss_body_clear)
        if self._ss_clear_hard_gate:
            ready_to_stack &= base_still_supported
        if bool(ready_to_stack.any()):
            ids = torch.nonzero(ready_to_stack, as_tuple=False).squeeze(-1)
            rows = top_rows[ids]
            self._ss_phase[ids] = self.STACK
            self._ss_clear_bonus_pending[ids] = True
            self._box_tar_pos[rows, 0:2] = base[ids, 0:2]
            base_size = self._box_lib._box_size[base_rows[ids]]
            top_size = self._box_lib._box_size[rows]
            self._box_tar_pos[rows, 2] = base[ids, 2] + 0.5 * (
                base_size[:, 2] + top_size[:, 2]
            )
            self._reset_steer(rows)
            self._mscale[rows] = 1.0
            self._compute_observations(ids)

        active = self._ss_phase == self.STACK
        if bool(active.any()):
            ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            rows = top_rows[ids]
            desired = self._box_tar_pos[rows].clone()
            desired[:, 0:2] = base[ids, 0:2]
            base_size = self._box_lib._box_size[base_rows[ids]]
            top_size = self._box_lib._box_size[rows]
            desired[:, 2] = base[ids, 2] + 0.5 * (base_size[:, 2] + top_size[:, 2])
            self._box_tar_pos[rows] = torch.lerp(self._box_tar_pos[rows], desired, 0.10)
            target = self._box_tar_pos[rows]
            top_ok = (
                ((top[ids, 0:2] - target[:, 0:2]).norm(dim=-1) <= self._ss_xy_tol)
                & ((top[ids, 2] - target[:, 2]).abs() <= self._ss_z_tol)
                & (top[ids, 7:10].norm(dim=-1) <= self._ss_stable_lin)
                & (top[ids, 10:13].norm(dim=-1) <= self._ss_stable_ang)
                & (self._upright_score(top[ids, 3:7]) >= math.cos(math.radians(10.0)))
            )
            self._ss_top_count[ids] = torch.where(
                top_ok,
                self._ss_top_count[ids] + 1,
                torch.zeros_like(self._ss_top_count[ids]),
            )
            success = ids[self._ss_top_count[ids] >= self._ss_top_steps]
            if len(success) > 0:
                self._ss_phase[success] = self.SUCCESS
                self._ss_success[success] = True
            self._compute_observations(ids)

        # No re-grasp controller: after release, losing the base ends the trial.
        post = (self._ss_phase >= self.RELEASE) & (self._ss_phase < self.SUCCESS)
        base_shift = (base[:, 0:2] - self._ss_base_goal[:, 0:2]).norm(dim=-1)
        dropped = post & ((base_shift > self._ss_drop_xy) | (base[:, 2] < -0.05))
        if bool(dropped.any()):
            ids = torch.nonzero(dropped, as_tuple=False).squeeze(-1)
            self._ss_phase[ids] = self.FAILED
            self._ss_failed[ids] = True
            self.reset_buf[ids] = 1

        self.extras["stack_phase"] = self._ss_phase.float()
        self.extras["stack_base_agent"] = self._ss_base_agent.float()
        self.extras["stack_hand_distance"] = hand_dist
        self.extras["stack_release_done"] = (self._ss_phase >= self.CLEAR).float()
        self.extras["stack_clear_done"] = (self._ss_phase >= self.STACK).float()
        self.extras["stack_staged"] = self._ss_staged.float()
        self.extras["stack_success"] = self._ss_success.float()
        self.extras["stack_failed"] = self._ss_failed.float()
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._update_stack_controller()
        return
