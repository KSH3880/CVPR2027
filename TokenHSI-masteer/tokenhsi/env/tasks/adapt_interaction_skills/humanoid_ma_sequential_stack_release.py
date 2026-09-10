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
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate

from env.tasks.adapt_interaction_skills.humanoid_ma_carry import (
    CARRY_HI, CARRY_LO, TEAMMATE_DIM,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp
from utils import torch_utils


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

    (
        DIAG_CARRY_STEPS,
        DIAG_ENTRY_XY,
        DIAG_ENTRY_Z,
        DIAG_ENTRY_LIN,
        DIAG_ENTRY_ANG,
        DIAG_ENTRY_UPRIGHT,
        DIAG_ENTRY_FOOT,
        DIAG_ENTRY_PLACEABLE,
        DIAG_ENTRY_JOINT,
        DIAG_ENTRY_STREAK_MAX,
        DIAG_RELEASE_STEPS,
        DIAG_RELEASE_HAND,
        DIAG_RELEASE_FOOT,
        DIAG_RELEASE_JOINT,
        DIAG_RELEASE_STREAK_MAX,
        DIAG_CLEAR_STEPS,
        DIAG_CLEAR_HAND,
        DIAG_CLEAR_BODY,
        DIAG_CLEAR_STABLE,
        DIAG_CLEAR_JOINT,
        DIAG_RETREAT_ROOT_DIST_MAX,
        DIAG_RETREAT_ARC_MAX,
        DIAG_CLEAR_FOOT,
    ) = range(23)
    DIAG_DIM = 23

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
        self._ss_stage_hand_tol = _f("STACK_STAGE_HAND_TOL", 0.35)
        self._ss_stage_stable_lin = _f("STACK_STAGE_STABLE_LIN", 0.25)
        self._ss_stage_stable_ang = _f("STACK_STAGE_STABLE_ANG", 1.0)
        self._ss_stage_hold_steps = _i("STACK_STAGE_HOLD_STEPS", 0)
        self._ss_require_staged = bool(_i("STACK_REQUIRE_STAGED", 0))
        self._ss_pre_steps = _i("STACK_PRE_STEPS", 0)
        self._ss_phase_steps = _i("STACK_PHASE_STEPS", 0)
        self._ss_body_clear = _f("STACK_BODY_CLEAR", 1.0)
        self._ss_retreat_dist = _f("STACK_RETREAT_DIST", 1.5)
        self._ss_retreat_side_deg = _f("STACK_RETREAT_SIDE_DEG", 0.0)
        # Keep the old implicit value as the class default so ms20 sidecars
        # remain replayable; the ms21 wrapper explicitly selects 0.5.
        self._ss_retreat_scale = _f("STACK_RETREAT_SCALE", 1.0)
        self._ss_clear_route_around = bool(_i("STACK_CLEAR_ROUTE_AROUND", 0))
        self._ss_clear_route_margin = _f("STACK_CLEAR_ROUTE_MARGIN", 0.30)
        self._ss_clear_stop_on_stack = bool(_i("STACK_CLEAR_STOP_ON_STACK", 0))

        self._ss_xy_tol = _f("STACK_XY_TOL", 0.10)
        self._ss_z_tol = _f("STACK_Z_TOL", 0.06)
        self._ss_entry_lin = _f("STACK_ENTRY_LIN", 0.25)
        self._ss_entry_ang = _f("STACK_ENTRY_ANG", 1.0)
        self._ss_entry_steps = _i("STACK_ENTRY_STEPS", 5)
        self._ss_entry_delivered = bool(_i("STACK_ENTRY_DELIVERED", 0))
        self._ss_entry_lowered = bool(_i("STACK_ENTRY_LOWERED", 0))
        self._ss_hand_clear = _f("STACK_HAND_CLEAR", 0.15)
        self._ss_hand_steps = _i("STACK_HAND_STEPS", 5)
        self._ss_release_carry_bridge = bool(_i("STACK_RELEASE_CARRY_BRIDGE", 0))
        self._ss_release_progress_w = _f("STACK_RELEASE_PROGRESS_W", 0.0)
        self._ss_release_hold_pen_w = _f("STACK_RELEASE_HOLD_PEN_W", 0.0)
        self._ss_release_hold_grace_steps = _i(
            "STACK_RELEASE_HOLD_GRACE_STEPS", 0
        )
        self._ss_success_bonus = _f("STACK_SUCCESS_BONUS", 0.0)
        self._ss_above_bonus = _f("STACK_ABOVE_BONUS", 0.0)
        self._ss_above_xy_tol = _f("STACK_ABOVE_XY_TOL", 0.20)
        self._ss_above_z_tol = _f("STACK_ABOVE_Z_TOL", 0.12)
        self._ss_top_scale = _f("STACK_TOP_SCALE", 1.0)
        self._ss_top_require_hand_clear = bool(_i(
            "STACK_TOP_REQUIRE_HAND_CLEAR", 0
        ))
        self._ss_top_hand_clear = _f("STACK_TOP_HAND_CLEAR", self._ss_hand_clear)
        self._ss_top_release_progress_w = _f(
            "STACK_TOP_RELEASE_PROGRESS_W", 0.0
        )
        self._ss_top_hold_pen_w = _f("STACK_TOP_HOLD_PEN_W", 0.0)
        self._ss_top_hold_grace_steps = _i("STACK_TOP_HOLD_GRACE_STEPS", 0)
        self._ss_top_approach_progress_w = _f(
            "STACK_TOP_APPROACH_PROGRESS_W", 0.0
        )
        self._ss_top_premature_release_pen_w = _f(
            "STACK_TOP_PREMATURE_RELEASE_PEN_W", 0.0
        )
        self._ss_top_settle_steps = _i("STACK_TOP_SETTLE_STEPS", 0)
        self._ss_hand_only_switch = bool(_i("STACK_HAND_ONLY_SWITCH", 0))
        self._ss_foot_clear = _f("STACK_FOOT_CLEAR", 0.20)
        self._ss_foot_box_w = _f("STACK_FOOT_BOX_W", 0.0)
        self._ss_carry_foot_gate = bool(_i("STACK_CARRY_FOOT_GATE", 0))
        self._ss_entry_foot_gate = bool(_i(
            "STACK_ENTRY_FOOT_GATE", int(self._ss_carry_foot_gate)
        ))
        self._ss_release_foot_gate = bool(_i(
            "STACK_RELEASE_FOOT_GATE", int(self._ss_carry_foot_gate)
        ))
        self._ss_carry_foot_xy = _f("STACK_CARRY_FOOT_XY", 0.30)
        self._ss_carry_foot_z = _f("STACK_CARRY_FOOT_Z", 0.20)
        self._ss_stable_lin = _f("STACK_STABLE_LIN", 0.08)
        self._ss_stable_ang = _f("STACK_STABLE_ANG", 0.20)
        self._ss_top_xy_tol = _f("STACK_TOP_XY_TOL", self._ss_xy_tol)
        self._ss_top_z_tol = _f("STACK_TOP_Z_TOL", self._ss_z_tol)
        self._ss_top_stable_lin = _f("STACK_TOP_STABLE_LIN", self._ss_stable_lin)
        self._ss_top_stable_ang = _f("STACK_TOP_STABLE_ANG", self._ss_stable_ang)
        self._ss_top_upright_deg = _f("STACK_TOP_UPRIGHT_DEG", 10.0)
        self._ss_top_steps = _i("STACK_TOP_STEPS", 20)
        self._ss_drop_xy = _f("STACK_DROP_XY", 0.25)
        self._ss_transition_bonus = _f("STACK_TRANSITION_BONUS", 3.0)
        self._ss_clear_bonus = _f("STACK_CLEAR_BONUS", 1.5)
        self._ss_clear_progress_w = _f("STACK_CLEAR_PROGRESS_W", 0.7)
        self._ss_clear_speed_w = _f("STACK_CLEAR_SPEED_W", 0.3)
        self._ss_clear_steer_w = _f("STACK_CLEAR_STEER_W", 0.25)
        self._ss_clear_vel_k = _f("STACK_CLEAR_VEL_K", 4.0)
        self._ss_clear_lat_k = _f("STACK_CLEAR_LAT_K", 1.0)
        self._ss_clear_min_speed = _f("STACK_CLEAR_MIN_SPEED", 0.05)
        self._ss_clear_hard_gate = bool(_i("STACK_CLEAR_HARD_GATE", 0))
        self._ss_clear_xy_tol = _f("STACK_CLEAR_XY_TOL", self._ss_xy_tol)
        self._ss_clear_z_tol = _f("STACK_CLEAR_Z_TOL", self._ss_z_tol)
        self._ss_clear_stable_lin = _f("STACK_CLEAR_STABLE_LIN", self._ss_stable_lin)
        self._ss_clear_stable_ang = _f("STACK_CLEAR_STABLE_ANG", self._ss_stable_ang)
        self._ss_clear_motion_gate = bool(_i("STACK_CLEAR_MOTION_GATE", 0))
        self._ss_clear_arc_dist = _f("STACK_CLEAR_ARC_DIST", 0.0)
        self._ss_clear_base_disp_w = _f("STACK_CLEAR_BASE_DISP_W", 0.0)
        self._ss_clear_base_lin_w = _f("STACK_CLEAR_BASE_LIN_W", 0.0)
        self._ss_clear_base_ang_w = _f("STACK_CLEAR_BASE_ANG_W", 0.0)
        self._ss_rehearsal_frac = _f("STACK_REHEARSAL_FRAC", 0.0)
        self._ss_virtual_retreat = bool(_i("STACK_VIRTUAL_RETREAT_BOX", 0))
        self._ss_virtual_rear_box = bool(
            _i("STACK_VIRTUAL_RETREAT_REAR_BOX", 0)
        )
        self._ss_zero_carry_obs = bool(_i("STACK_ZERO_CARRY_OBS", 0))
        self._ss_dynamic_carry_mask = bool(_i("STACK_DYNAMIC_CARRY_MASK", 0))
        self._ss_clear_signed = bool(_i("STACK_CLEAR_SIGNED", 0))
        self._ss_sequential_reward_mask = bool(_i("STACK_SEQUENTIAL_REWARD_MASK", 0))
        self._ss_negative_clear_reward = bool(
            _i("STACK_NEGATIVE_CLEAR_REWARD", 0)
        )
        self._ss_clear_move_w = _f("STACK_CLEAR_MOVE_W", 1.0)
        self._ss_clear_hand_pen_w = _f("STACK_CLEAR_HAND_PEN_W", 0.5)
        self._ss_clear_stall_pen_w = _f("STACK_CLEAR_STALL_PEN_W", 0.5)
        self._ss_clear_reverse_pen_w = _f("STACK_CLEAR_REVERSE_PEN_W", 0.5)
        self._ss_clear_move_min_frac = _f("STACK_CLEAR_MOVE_MIN_FRAC", 0.20)
        self._ss_clear_grace_steps = _i("STACK_CLEAR_GRACE_STEPS", 0)
        self._ss_carry_done_r = _f("STACK_CARRY_DONE_REWARD", 1.6)
        self._ss_release_done_r = _f("STACK_RELEASE_DONE_REWARD", 1.0)
        self._ss_clear_done_r = _f(
            "STACK_CLEAR_DONE_REWARD", 1.0 + self._ss_clear_steer_w
        )
        if self._ss_retreat_scale <= 0.0 or self._ss_retreat_scale > 1.0:
            raise ValueError("STACK_RETREAT_SCALE must be in (0, 1]")
        if self._ss_retreat_side_deg < 0.0 or self._ss_retreat_side_deg > 90.0:
            raise ValueError("STACK_RETREAT_SIDE_DEG must be in [0, 90]")
        if self._ss_rehearsal_frac < 0.0 or self._ss_rehearsal_frac > 1.0:
            raise ValueError("STACK_REHEARSAL_FRAC must be in [0, 1]")
        if self._ss_clear_progress_w < 0.0 or self._ss_clear_speed_w < 0.0:
            raise ValueError("STACK_CLEAR_PROGRESS_W/STACK_CLEAR_SPEED_W must be non-negative")
        if self._ss_clear_steer_w < 0.0:
            raise ValueError("STACK_CLEAR_STEER_W must be non-negative")
        if self._ss_release_progress_w < 0.0 or self._ss_release_hold_pen_w < 0.0:
            raise ValueError(
                "STACK_RELEASE_PROGRESS_W/STACK_RELEASE_HOLD_PEN_W "
                "must be non-negative"
            )
        if self._ss_release_hold_grace_steps < 0:
            raise ValueError("STACK_RELEASE_HOLD_GRACE_STEPS must be non-negative")
        if self._ss_success_bonus < 0.0:
            raise ValueError("STACK_SUCCESS_BONUS must be non-negative")
        if self._ss_above_bonus < 0.0:
            raise ValueError("STACK_ABOVE_BONUS must be non-negative")
        if self._ss_clear_route_margin < 0.0:
            raise ValueError("STACK_CLEAR_ROUTE_MARGIN must be non-negative")
        if self._ss_top_scale <= 0.0 or self._ss_top_scale > 1.0:
            raise ValueError("STACK_TOP_SCALE must be in (0, 1]")
        if min(
            self._ss_above_xy_tol,
            self._ss_above_z_tol,
            self._ss_top_xy_tol,
            self._ss_top_z_tol,
            self._ss_top_stable_lin,
            self._ss_top_stable_ang,
            self._ss_top_hand_clear,
        ) <= 0.0:
            raise ValueError("STACK top/above tolerances must be positive")
        if self._ss_top_upright_deg <= 0.0 or self._ss_top_upright_deg > 90.0:
            raise ValueError("STACK_TOP_UPRIGHT_DEG must be in (0, 90]")
        if self._ss_top_release_progress_w < 0.0 or self._ss_top_hold_pen_w < 0.0:
            raise ValueError("STACK top release weights must be non-negative")
        if self._ss_top_hold_grace_steps < 0:
            raise ValueError("STACK_TOP_HOLD_GRACE_STEPS must be non-negative")
        if min(
            self._ss_stage_hand_tol,
            self._ss_stage_stable_lin,
            self._ss_stage_stable_ang,
        ) <= 0.0:
            raise ValueError("STACK stage hand/stability tolerances must be positive")
        if min(
            self._ss_stage_hold_steps,
            self._ss_pre_steps,
            self._ss_phase_steps,
            self._ss_top_settle_steps,
        ) < 0:
            raise ValueError("STACK stage/time/settle steps must be non-negative")
        if self._ss_require_staged and self._ss_stage_hold_steps <= 0:
            raise ValueError(
                "STACK_REQUIRE_STAGED requires STACK_STAGE_HOLD_STEPS > 0"
            )
        if (self._ss_pre_steps == 0) != (self._ss_phase_steps == 0):
            raise ValueError(
                "STACK_PRE_STEPS and STACK_PHASE_STEPS must both be zero or positive"
            )
        if min(
            self._ss_top_approach_progress_w,
            self._ss_top_premature_release_pen_w,
        ) < 0.0:
            raise ValueError("STACK top approach/release weights must be non-negative")
        if self._ss_release_carry_bridge and (
            self._ss_release_progress_w > 0.0
            or self._ss_release_hold_pen_w > 0.0
        ):
            raise ValueError(
                "STACK_RELEASE_CARRY_BRIDGE cannot be combined with isolated "
                "RELEASE progress/hold shaping"
            )
        if min(self._ss_carry_done_r, self._ss_release_done_r,
               self._ss_clear_done_r) < 0.0:
            raise ValueError("completed task rewards must be non-negative")
        if self._ss_dynamic_carry_mask and self._ss_virtual_retreat:
            raise ValueError(
                "STACK_DYNAMIC_CARRY_MASK and STACK_VIRTUAL_RETREAT_BOX "
                "cannot both be enabled"
            )
        if self._ss_zero_carry_obs and self._ss_virtual_retreat:
            raise ValueError(
                "STACK_ZERO_CARRY_OBS and STACK_VIRTUAL_RETREAT_BOX "
                "cannot both be enabled"
            )
        if self._ss_virtual_rear_box and not self._ss_virtual_retreat:
            raise ValueError(
                "STACK_VIRTUAL_RETREAT_REAR_BOX requires "
                "STACK_VIRTUAL_RETREAT_BOX"
            )
        if min(self._ss_clear_arc_dist, self._ss_clear_base_disp_w,
               self._ss_clear_base_lin_w, self._ss_clear_base_ang_w) < 0.0:
            raise ValueError("CLEAR arc distance and base penalty weights must be non-negative")
        if min(self._ss_clear_move_w, self._ss_clear_hand_pen_w,
               self._ss_clear_stall_pen_w, self._ss_clear_reverse_pen_w) < 0.0:
            raise ValueError("negative CLEAR reward weights must be non-negative")
        if self._ss_clear_move_min_frac <= 0.0 or self._ss_clear_move_min_frac > 1.0:
            raise ValueError("STACK_CLEAR_MOVE_MIN_FRAC must be in (0, 1]")
        if self._ss_clear_grace_steps < 0:
            raise ValueError("STACK_CLEAR_GRACE_STEPS must be non-negative")
        if self._ss_foot_clear <= 0.0 or self._ss_foot_box_w < 0.0:
            raise ValueError("STACK_FOOT_CLEAR must be positive and STACK_FOOT_BOX_W non-negative")
        if self._ss_carry_foot_xy <= 0.0 or self._ss_carry_foot_z <= 0.0:
            raise ValueError("STACK_CARRY_FOOT_XY/Z must be positive")
        if min(self._ss_clear_xy_tol, self._ss_clear_z_tol,
               self._ss_clear_stable_lin, self._ss_clear_stable_ang) <= 0.0:
            raise ValueError("CLEAR gate tolerances must be positive")
        clear_w = self._ss_clear_progress_w + self._ss_clear_speed_w
        if clear_w <= 0.0:
            raise ValueError("at least one CLEAR motion reward weight must be positive")
        self._ss_clear_progress_w /= clear_w
        self._ss_clear_speed_w /= clear_w

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self._ss_phase_steps > 0 and self.max_episode_length < (
            self._ss_pre_steps + self._ss_phase_steps
        ):
            raise ValueError(
                "episodeLength must cover STACK_PRE_STEPS + STACK_PHASE_STEPS"
            )

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
        self._ss_stage_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_top_place_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_top_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ss_staged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_top_settled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_bonus_pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_clear_bonus_pending = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_success_bonus_pending = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_above_bonus_pending = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_above_seen = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ss_base_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_stage_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_top_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_latched_base = torch.zeros((num_envs, 3), device=self.device)
        self._ss_retreat_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_retreat_dir = torch.zeros((num_envs, 2), device=self.device)
        self._ss_clear_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_release_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_prev_release_h = torch.zeros(num_envs, device=self.device)
        self._ss_top_release_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_stack_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_prev_top_h = torch.zeros(num_envs, device=self.device)
        self._ss_prev_top_dist = torch.zeros(num_envs, device=self.device)
        rehearsal = np.zeros(num_envs, dtype=np.bool_)
        rehearsal_count = int(round(num_envs * self._ss_rehearsal_frac))
        if rehearsal_count:
            order = np.arange(num_envs)
            np.random.default_rng(self._ss_seed + 1).shuffle(order)
            rehearsal[order[:rehearsal_count]] = True
        self._ss_rehearsal = torch.as_tensor(
            rehearsal, dtype=torch.bool, device=self.device
        )
        rows = self._rows
        self._ss_dbg_z0 = torch.zeros(rows, device=self.device)
        self._ss_dbg_max_lift = torch.zeros(rows, device=self.device)
        self._ss_dbg_base_disp_max = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_gate_counts = torch.zeros(
            (num_envs, self.DIAG_DIM), device=self.device
        )
        self._ss_dbg_near_step = torch.full(
            (rows,), -1, dtype=torch.long, device=self.device
        )
        self._ss_dbg_pick_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_break_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_place_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_release_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_clear_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_success_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_fail_step = torch.full_like(self._ss_dbg_near_step, -1)
        self._ss_dbg_staged_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._ss_dbg_settled_step = torch.full_like(self._ss_dbg_staged_step, -1)
        self._ss_dbg_above_step = torch.full_like(self._ss_dbg_staged_step, -1)
        self._ss_dbg_stack_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_dbg_stack_grasp_steps = torch.zeros_like(self._ss_dbg_stack_steps)
        self._ss_dbg_stack_place_steps = torch.zeros_like(self._ss_dbg_stack_steps)
        self._ss_dbg_stack_dist_start = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_stack_dist_min = torch.full(
            (num_envs,), float("inf"), device=self.device
        )
        self._ss_dbg_top_native_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_top_approach_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_top_premature_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_top_release_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_top_hold_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_above_bonus_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_success_bonus_sum = torch.zeros(num_envs, device=self.device)
        self._ss_dbg_end_reason = torch.full_like(self._ss_dbg_staged_step, -1)
        self._ss_dbg_pick_count = torch.zeros(
            rows, dtype=torch.long, device=self.device
        )
        self._ss_dbg_lost_count = torch.zeros_like(self._ss_dbg_pick_count)
        print(
            "[stack-release] shared policy; balanced random base/top roles; "
            f"transition_bonus={self._ss_transition_bonus:.2f} "
            f"clear_bonus={self._ss_clear_bonus:.2f} "
            f"clear_steer_w={self._ss_clear_steer_w:.2f} "
            f"retreat_scale={self._ss_retreat_scale:.2f} "
            f"retreat_side_deg={self._ss_retreat_side_deg:.1f} "
            f"hand_only={int(self._ss_hand_only_switch)} "
            f"foot_box_w={self._ss_foot_box_w:.2f} "
            f"carry_foot_gate={int(self._ss_carry_foot_gate)} "
            f"entry_foot_gate={int(self._ss_entry_foot_gate)} "
            f"entry_delivered={int(self._ss_entry_delivered)} "
            f"entry_lowered={int(self._ss_entry_lowered)} "
            f"release_foot_gate={int(self._ss_release_foot_gate)} "
            f"release_carry_bridge={int(self._ss_release_carry_bridge)} "
            f"clear_route={int(self._ss_clear_route_around)}/"
            f"{self._ss_clear_route_margin:.2f}/"
            f"{int(self._ss_clear_stop_on_stack)} "
            f"stage={self._ss_stage_dist:.2f}m/"
            f"{self._ss_stage_hold_steps}/req{int(self._ss_require_staged)} "
            f"budget={self._ss_pre_steps}+{self._ss_phase_steps} "
            f"top_scale={self._ss_top_scale:.2f} "
            f"above_bonus={self._ss_above_bonus:.2f} "
            f"top_release={self._ss_top_release_progress_w:.2f}/"
            f"{self._ss_top_hold_pen_w:.2f}/"
            f"{self._ss_top_hold_grace_steps} "
            f"top_approach={self._ss_top_approach_progress_w:.2f} "
            f"top_premature={self._ss_top_premature_release_pen_w:.2f} "
            f"top_settle={self._ss_top_settle_steps} "
            f"release_progress_w={self._ss_release_progress_w:.2f} "
            f"release_hold={self._ss_release_hold_pen_w:.2f}/"
            f"{self._ss_release_hold_grace_steps} "
            f"success_bonus={self._ss_success_bonus:.2f} "
            f"sequential_reward_mask={int(self._ss_sequential_reward_mask)} "
            f"done_r={self._ss_carry_done_r:.2f}/"
            f"{self._ss_release_done_r:.2f}/{self._ss_clear_done_r:.2f} "
            f"clear_arc={self._ss_clear_arc_dist:.2f} "
            f"clear_base_w={self._ss_clear_base_disp_w:.2f}/"
            f"{self._ss_clear_base_lin_w:.2f}/{self._ss_clear_base_ang_w:.2f} "
            f"clear_motion_gate={int(self._ss_clear_motion_gate)} "
            f"rehearsal={self._ss_rehearsal_frac:.2f} "
            f"virtual_retreat={int(self._ss_virtual_retreat)} "
            f"virtual_rear_box={int(self._ss_virtual_rear_box)} "
            f"zero_carry_obs={int(self._ss_zero_carry_obs)} "
            f"dynamic_carry_mask={int(self._ss_dynamic_carry_mask)} "
            f"negative_clear_reward={int(self._ss_negative_clear_reward)} "
            f"negative_clear_w={self._ss_clear_move_w:.2f}/"
            f"{self._ss_clear_hand_pen_w:.2f}/"
            f"{self._ss_clear_stall_pen_w:.2f}/"
            f"{self._ss_clear_reverse_pen_w:.2f} "
            f"move_min_frac={self._ss_clear_move_min_frac:.2f} "
            f"clear_grace={self._ss_clear_grace_steps}",
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

    def _reset_stack_debug(self, env_ids):
        for value in (
            self._ss_dbg_staged_step,
            self._ss_dbg_settled_step,
            self._ss_dbg_above_step,
            self._ss_dbg_end_reason,
        ):
            value[env_ids] = -1
        for value in (
            self._ss_dbg_stack_steps,
            self._ss_dbg_stack_grasp_steps,
            self._ss_dbg_stack_place_steps,
            self._ss_dbg_stack_dist_start,
            self._ss_dbg_top_native_sum,
            self._ss_dbg_top_approach_sum,
            self._ss_dbg_top_premature_sum,
            self._ss_dbg_top_release_sum,
            self._ss_dbg_top_hold_sum,
            self._ss_dbg_above_bonus_sum,
            self._ss_dbg_success_bonus_sum,
        ):
            value[env_ids] = 0
        self._ss_dbg_stack_dist_min[env_ids] = float("inf")

    def _post_object_reset(self, env_ids):
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return
        rows = self.agent_rows(env_ids)
        self._reset_stage_metrics(rows)
        self._ss_dbg_z0[rows] = self.humanoid_rows(self._box_states)[rows, 2]
        self._ss_phase[env_ids] = self.CARRY
        self._ss_entry_count[env_ids] = 0
        self._ss_hand_count[env_ids] = 0
        self._ss_stage_count[env_ids] = 0
        self._ss_top_place_count[env_ids] = 0
        self._ss_top_count[env_ids] = 0
        self._ss_staged[env_ids] = False
        self._ss_top_settled[env_ids] = False
        self._ss_bonus_pending[env_ids] = False
        self._ss_clear_bonus_pending[env_ids] = False
        self._ss_success_bonus_pending[env_ids] = False
        self._ss_above_bonus_pending[env_ids] = False
        self._ss_above_seen[env_ids] = False
        self._ss_top_release_age[env_ids] = 0
        self._ss_stack_age[env_ids] = 0
        self._ss_prev_top_h[env_ids] = 0.0
        self._ss_prev_top_dist[env_ids] = 0.0
        self._ss_success[env_ids] = False
        self._ss_failed[env_ids] = False
        self._ss_latched_base[env_ids] = 0.0
        self._ss_retreat_goal[env_ids] = 0.0
        self._ss_retreat_dir[env_ids] = 0.0
        self._ss_clear_age[env_ids] = 0
        self._ss_release_age[env_ids] = 0
        self._ss_prev_release_h[env_ids] = 0.0
        self._ss_dbg_base_disp_max[env_ids] = 0.0
        self._ss_dbg_gate_counts[env_ids] = 0.0
        self._reset_stack_debug(env_ids)

        env_ids = env_ids[~self._ss_rehearsal[env_ids]]
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
        self._ss_stage_count[env_ids] = 0
        self._ss_top_place_count[env_ids] = 0
        self._ss_top_count[env_ids] = 0
        self._ss_staged[env_ids] = False
        self._ss_top_settled[env_ids] = False
        self._ss_bonus_pending[env_ids] = False
        self._ss_clear_bonus_pending[env_ids] = False
        self._ss_success_bonus_pending[env_ids] = False
        self._ss_success[env_ids] = False
        self._ss_failed[env_ids] = False
        self._ss_latched_base[env_ids] = 0.0
        self._ss_release_age[env_ids] = 0
        self._ss_prev_release_h[env_ids] = 0.0

        self._sync_stack_actors(env_ids)
        rows = self.agent_rows(env_ids)
        self._ss_dbg_z0[rows] = self.humanoid_rows(self._box_states)[rows, 2]
        self._reset_steer(self.agent_rows(env_ids))
        self._mscale[base_rows] = 1.0
        self._mscale[top_rows] = 1.0
        self._reset_progress_ref(env_ids)
        return

    # ------------------------------------------------------------- measurements

    def _hand_surface_distances(self, rows):
        box = self.humanoid_rows(self._box_states)[rows]
        rb = self.humanoid_rows(self._rigid_body_pos)[rows]
        hands = rb[:, self._key_body_ids[[0, 1]], :]
        rel = hands - box[:, None, 0:3]
        inv = quat_conjugate(box[:, 3:7])[:, None, :].expand(-1, 2, -1)
        local = quat_rotate(inv.reshape(-1, 4), rel.reshape(-1, 3)).view(-1, 2, 3)
        half = 0.5 * self._box_lib._box_size[rows, None, :]
        outside = torch.clamp(local.abs() - half, min=0.0)
        return outside.norm(dim=-1)

    def _hand_surface_distance(self, rows):
        # Minimum over hands: both hands must clear the threshold.
        return self._hand_surface_distances(rows).amin(dim=-1)

    def _foot_surface_distance(self, rows):
        box = self.humanoid_rows(self._box_states)[rows]
        rb = self.humanoid_rows(self._rigid_body_pos)[rows]
        feet = rb[:, self._key_body_ids[[2, 3]], :]
        rel = feet - box[:, None, 0:3]
        inv = quat_conjugate(box[:, 3:7])[:, None, :].expand(-1, 2, -1)
        local = quat_rotate(inv.reshape(-1, 4), rel.reshape(-1, 3)).view(-1, 2, 3)
        half = 0.5 * self._box_lib._box_size[rows, None, :]
        outside = torch.clamp(local.abs() - half, min=0.0)
        return outside.norm(dim=-1).amin(dim=-1)

    def _all_foot_surface_distance(self, base_rows, top_rows):
        box = self.humanoid_rows(self._box_states)[base_rows]
        rb = self.humanoid_rows(self._rigid_body_pos)
        feet = torch.cat(
            (
                rb[base_rows][:, self._key_body_ids[[2, 3]], :],
                rb[top_rows][:, self._key_body_ids[[2, 3]], :],
            ),
            dim=1,
        )
        rel = feet - box[:, None, 0:3]
        inv = quat_conjugate(box[:, 3:7])[:, None, :].expand(-1, 4, -1)
        local = quat_rotate(inv.reshape(-1, 4), rel.reshape(-1, 3)).view(-1, 4, 3)
        half = 0.5 * self._box_lib._box_size[base_rows, None, :]
        outside = torch.clamp(local.abs() - half, min=0.0)
        return outside.norm(dim=-1).amin(dim=-1)

    def _reset_stage_metrics(self, rows):
        for value in (
            self._ss_dbg_near_step,
            self._ss_dbg_pick_step,
            self._ss_dbg_break_step,
            self._ss_dbg_place_step,
            self._ss_dbg_release_step,
            self._ss_dbg_clear_step,
            self._ss_dbg_success_step,
            self._ss_dbg_fail_step,
        ):
            value[rows] = -1
        self._ss_dbg_max_lift[rows] = 0.0
        self._ss_dbg_pick_count[rows] = 0
        self._ss_dbg_lost_count[rows] = 0
        return

    def _update_stage_metrics(self):
        rows = self.all_rows()
        box_z = self.humanoid_rows(self._box_states)[rows, 2]
        lift = box_z - self._ss_dbg_z0[rows]
        self._ss_dbg_max_lift[rows] = torch.maximum(
            self._ss_dbg_max_lift[rows], lift
        )
        both_near = self._hand_surface_distances(rows).amax(dim=-1) <= 0.35
        step = self.progress_rows()

        first_near = both_near & (self._ss_dbg_near_step < 0)
        self._ss_dbg_near_step[first_near] = step[first_near]

        pick_now = both_near & (lift >= 0.10)
        self._ss_dbg_pick_count = torch.where(
            pick_now,
            self._ss_dbg_pick_count + 1,
            torch.zeros_like(self._ss_dbg_pick_count),
        )
        first_pick = (
            (self._ss_dbg_pick_count >= 10) & (self._ss_dbg_pick_step < 0)
        )
        self._ss_dbg_pick_step[first_pick] = step[first_pick]

        carrying = (self._ss_dbg_pick_step >= 0) & (self._ep_finish < 0)
        lost_now = carrying & ((~both_near) | (lift < 0.05))
        self._ss_dbg_lost_count = torch.where(
            lost_now,
            self._ss_dbg_lost_count + 1,
            torch.zeros_like(self._ss_dbg_lost_count),
        )
        first_break = (
            (self._ss_dbg_lost_count >= 10) & (self._ss_dbg_break_step < 0)
        )
        self._ss_dbg_break_step[first_break] = step[first_break]

        env_ids = torch.arange(self.num_envs, device=self.device)
        base_rows, _ = self._role_rows(env_ids)
        base_xy = self.humanoid_rows(self._box_states)[base_rows, 0:2]
        base_disp = (base_xy - self._ss_latched_base[:, 0:2]).norm(dim=-1)
        post_place = (
            (self._ss_phase >= self.RELEASE) & (self._ss_phase <= self.SUCCESS)
        )
        self._ss_dbg_base_disp_max = torch.where(
            post_place,
            torch.maximum(self._ss_dbg_base_disp_max, base_disp),
            self._ss_dbg_base_disp_max,
        )
        return

    def _metric_extra_cols(self, rows):
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        agent = rows % self.num_agents
        kind = (agent == self._ss_base_agent[env]).float()
        kind = torch.where(self._ss_rehearsal[env], -torch.ones_like(kind), kind)
        base_disp = torch.where(
            kind == 1,
            self._ss_dbg_base_disp_max[env],
            torch.zeros_like(kind),
        )
        gate_counts = torch.where(
            (kind == 1)[:, None],
            self._ss_dbg_gate_counts[env],
            torch.zeros_like(self._ss_dbg_gate_counts[env]),
        )
        base_role = kind == 1

        def base_debug(value):
            return torch.where(base_role, value[env], torch.zeros_like(value[env]))

        stack_dist_min = torch.where(
            torch.isfinite(self._ss_dbg_stack_dist_min),
            self._ss_dbg_stack_dist_min,
            -torch.ones_like(self._ss_dbg_stack_dist_min),
        )
        return super()._metric_extra_cols(rows) + [
            self._ss_dbg_near_step[rows].float(),
            self._ss_dbg_pick_step[rows].float(),
            self._ss_dbg_break_step[rows].float(),
            self._ss_dbg_place_step[rows].float(),
            self._ss_dbg_release_step[rows].float(),
            self._ss_dbg_clear_step[rows].float(),
            self._ss_dbg_fail_step[rows].float(),
            self._ss_dbg_max_lift[rows],
            kind,
            base_disp,
        ] + [gate_counts[:, index] for index in range(self.DIAG_DIM)] + [
            self._ss_dbg_success_step[rows].float(),
            base_debug(self._ss_dbg_staged_step).float(),
            base_debug(self._ss_dbg_settled_step).float(),
            base_debug(self._ss_dbg_above_step).float(),
            base_debug(self._ss_dbg_stack_steps).float(),
            base_debug(self._ss_dbg_stack_grasp_steps).float(),
            base_debug(self._ss_dbg_stack_place_steps).float(),
            base_debug(self._ss_dbg_stack_dist_start),
            base_debug(stack_dist_min),
            base_debug(self._ss_dbg_top_native_sum),
            base_debug(self._ss_dbg_top_approach_sum),
            base_debug(self._ss_dbg_top_premature_sum),
            base_debug(self._ss_dbg_top_release_sum),
            base_debug(self._ss_dbg_top_hold_sum),
            base_debug(self._ss_dbg_above_bonus_sum),
            base_debug(self._ss_dbg_success_bonus_sum),
            base_debug(self._ss_dbg_end_reason).float(),
        ]

    def _metric_reset_extra(self, rows):
        super()._metric_reset_extra(rows)
        self._reset_stage_metrics(rows)
        env = torch.div(rows, self.num_agents, rounding_mode="floor").unique()
        self._ss_dbg_base_disp_max[env] = 0.0
        self._ss_dbg_gate_counts[env] = 0.0
        self._reset_stack_debug(env)
        return

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
        # ``extras`` persists across steps.  CARRY diagnostics are conditional
        # on non-rehearsal base rows that have already picked the box, so reset
        # them before any early return and let the observer ignore NaNs.
        nan = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self.extras["tb/stack_reward/success_bonus"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.extras["tb/stack_reward/above_bonus"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for key in (
            "top_native", "top_approach_progress",
            "top_premature_release_penalty", "top_release_progress",
            "top_hold_penalty", "top_total",
        ):
            self.extras[f"tb/stack_reward/{key}"] = nan
        for key in (
            "top_distance", "top_above", "top_grasp_ok", "top_settled",
            "top_release_ready", "top_hand_factor", "top_grasp_factor", "top_hand_clear",
            "top_stack_age",
        ):
            self.extras[f"tb/stack_state/{key}"] = nan
        for key in ("carry_native", "carry_foot_penalty"):
            self.extras[f"tb/stack_reward/{key}"] = nan
        for key in (
            "release_total", "release_local", "release_hand_positive",
            "release_support_positive", "release_foot_penalty",
            "release_transition_bonus", "release_progress",
            "release_hold_penalty",
        ):
            self.extras[f"tb/stack_reward/{key}"] = nan
        for key in (
            "carry_target_dist", "carry_grasp_ok", "carry_break_rate",
            "near_goal_not_delivered", "carry_path_fraction",
        ):
            self.extras[f"tb/stack_state/{key}"] = nan
        for key in (
            "release_hand_factor", "release_hand_clear_rate",
            "release_streak_fraction", "release_support", "release_stable",
            "release_foot_distance", "release_age",
        ):
            self.extras[f"tb/stack_state/{key}"] = nan
        if self._ss_negative_clear_reward:
            # ``extras`` persists across steps; overwrite all phase-conditional
            # diagnostics with NaN before any early return to avoid stale values.
            for key in (
                "clear_total", "move_positive", "hand_penalty",
                "stall_penalty", "reverse_penalty", "base_penalty",
                "foot_penalty", "transition_bonus",
            ):
                self.extras[f"tb/stack_reward/{key}"] = nan
            for key in (
                "hand_factor", "hand_clear_rate", "recontact_rate",
                "move_ratio", "move_ok_rate", "stall_rate", "reverse_rate",
                "v_along", "v_command", "retreat_arc",
            ):
                self.extras[f"tb/stack_state/{key}"] = nan
        post = (self._ss_phase >= self.RELEASE) & (self._ss_phase <= self.SUCCESS)
        failed = self._ss_phase == self.FAILED
        use_carry_foot = self._ss_carry_foot_gate and self._ss_foot_box_w > 0.0
        top_debug = (
            self._ss_require_staged
            or self._ss_top_approach_progress_w > 0.0
            or self._ss_top_premature_release_pen_w > 0.0
            or self._ss_top_settle_steps > 0
        )
        if not (
            bool(post.any()) or bool(failed.any()) or use_carry_foot or top_debug
        ):
            return

        (base_rows, top_rows, base, top, hand_dist, xy_err, z_err, support,
         stable, root_dist) = self._base_features()
        carry_r = self.rew_buf[base_rows].clone()
        top_native = self.rew_buf[top_rows].clone()
        top_target = self._box_tar_pos[top_rows]
        top_xy_err = (top[:, 0:2] - top_target[:, 0:2]).norm(dim=-1)
        top_z_err = (top[:, 2] - top_target[:, 2]).abs()
        top_dist = torch.sqrt(top_xy_err.square() + top_z_err.square())
        stack_phase = self._ss_phase == self.STACK
        top_above = (
            stack_phase
            & (top_xy_err <= self._ss_above_xy_tol)
            & (top_z_err <= self._ss_above_z_tol)
        )
        top_hand_surfaces = self._hand_surface_distances(top_rows)
        top_hand_dist = top_hand_surfaces.amin(dim=-1)
        top_h = torch.clamp(top_hand_dist / self._ss_top_hand_clear, 0.0, 1.0)
        # Unlike final hand-clear, holding is broken as soon as either hand
        # separates.  This prevents the policy from starting a one-hand
        # release during WAIT while still receiving the old carry reward.
        top_grasp_factor = torch.clamp(
            top_hand_surfaces.amax(dim=-1) / self._ss_stage_hand_tol,
            0.0,
            1.0,
        )
        top_grasp_ok = (
            top_hand_surfaces.amax(dim=-1) <= self._ss_stage_hand_tol
        )
        release_ready = stack_phase & self._ss_top_settled
        top_approach_progress = (
            self._ss_top_approach_progress_w
            * (self._ss_prev_top_dist - top_dist)
            * stack_phase.float()
        )
        top_picked = self._ss_dbg_pick_step[top_rows] >= 0
        top_hold_required = (
            (~self._ss_rehearsal)
            & (self._ss_phase <= self.STACK)
            & top_picked
            & (~release_ready)
        )
        top_premature_release_penalty = (
            self._ss_top_premature_release_pen_w
            * top_grasp_factor
            * top_hold_required.float()
        )
        top_release_progress = (
            self._ss_top_release_progress_w
            * (top_h - self._ss_prev_top_h)
            * release_ready.float()
        )
        top_hold_penalty = (
            self._ss_top_hold_pen_w
            * (1.0 - top_h)
            * (self._ss_top_release_age >= self._ss_top_hold_grace_steps).float()
            * release_ready.float()
        )
        above_bonus = self._ss_above_bonus_pending.float() * self._ss_above_bonus
        top_local = (
            top_approach_progress
            - top_premature_release_penalty
            + top_release_progress
            - top_hold_penalty
            + above_bonus
        )
        self.rew_buf[top_rows] += top_local
        self._ss_prev_top_dist = torch.where(
            stack_phase, top_dist, torch.zeros_like(self._ss_prev_top_dist)
        )
        self._ss_prev_top_h = torch.where(
            release_ready, top_h, torch.zeros_like(self._ss_prev_top_h)
        )
        top_sample = top_hold_required | stack_phase | self._ss_above_bonus_pending
        self._ss_dbg_top_native_sum += top_native * stack_phase.float()
        self._ss_dbg_top_approach_sum += top_approach_progress
        self._ss_dbg_top_premature_sum += top_premature_release_penalty
        self._ss_dbg_top_release_sum += top_release_progress
        self._ss_dbg_top_hold_sum += top_hold_penalty
        self._ss_dbg_above_bonus_sum += above_bonus
        self.extras["tb/stack_reward/above_bonus"] = above_bonus
        for key, value in (
            ("top_native", top_native),
            ("top_approach_progress", top_approach_progress),
            ("top_premature_release_penalty", -top_premature_release_penalty),
            ("top_release_progress", top_release_progress),
            ("top_hold_penalty", -top_hold_penalty),
            ("top_total", self.rew_buf[top_rows]),
        ):
            self.extras[f"tb/stack_reward/{key}"] = torch.where(
                top_sample, value, nan
            )
        for key, value in (
            ("top_distance", top_dist),
            ("top_above", top_above.float()),
            ("top_grasp_ok", top_grasp_ok.float()),
            ("top_settled", self._ss_top_settled.float()),
            ("top_release_ready", release_ready.float()),
            ("top_hand_factor", top_h),
            ("top_grasp_factor", top_grasp_factor),
            ("top_hand_clear", (top_hand_dist >= self._ss_top_hand_clear).float()),
            ("top_stack_age", self._ss_stack_age.float()),
        ):
            self.extras[f"tb/stack_state/{key}"] = torch.where(
                top_sample, value, nan
            )
        foot_dist = (
            self._all_foot_surface_distance(base_rows, top_rows)
            if self._ss_carry_foot_gate
            else self._foot_surface_distance(base_rows)
        )
        foot_penalty = torch.clamp(
            1.0 - foot_dist / self._ss_foot_clear, 0.0, 1.0
        )
        carry_foot_active = (
            (self._ss_phase == self.CARRY)
            & (self._ss_dbg_pick_step[base_rows] >= 0)
            & (xy_err <= self._ss_carry_foot_xy)
            & (z_err <= self._ss_carry_foot_z)
        )
        carry_sample = (
            (self._ss_phase == self.CARRY)
            & (~self._ss_rehearsal)
            & (self._ss_dbg_pick_step[base_rows] >= 0)
        )

        def tb_carry(value):
            return torch.where(carry_sample, value, nan)

        target_dist = torch.sqrt(xy_err.square() + z_err.square())
        grasp_ok = (
            self._hand_surface_distances(base_rows).amax(dim=-1) <= 0.35
        )
        delivered = self._ep_finish[base_rows] >= 0
        path_fraction = self._arc_root[base_rows] / self._s_end[base_rows].clamp(
            min=1e-6
        )
        self.extras["tb/stack_reward/carry_native"] = tb_carry(carry_r)
        self.extras["tb/stack_reward/carry_foot_penalty"] = tb_carry(
            -self._ss_foot_box_w
            * foot_penalty
            * carry_foot_active.float()
        )
        self.extras["tb/stack_state/carry_target_dist"] = tb_carry(target_dist)
        self.extras["tb/stack_state/carry_grasp_ok"] = tb_carry(grasp_ok.float())
        self.extras["tb/stack_state/carry_break_rate"] = tb_carry(
            (self._ss_dbg_break_step[base_rows] >= 0).float()
        )
        self.extras["tb/stack_state/near_goal_not_delivered"] = tb_carry(
            ((target_dist <= 0.50) & (~delivered)).float()
        )
        self.extras["tb/stack_state/carry_path_fraction"] = tb_carry(
            path_fraction
        )
        if use_carry_foot and bool(carry_foot_active.any()):
            rows = base_rows[carry_foot_active]
            self.rew_buf[rows] = self.rew_buf[rows] - (
                self._ss_foot_box_w * foot_penalty[carry_foot_active]
            )

        self.extras["stack_foot_box_distance"] = foot_dist
        self.extras["stack_foot_box_penalty"] = foot_penalty
        if not (bool(post.any()) or bool(failed.any())):
            return

        h = torch.clamp(hand_dist / self._ss_hand_clear, 0.0, 1.0)
        clear_now = hand_dist >= self._ss_hand_clear
        # CLEAR uses the retreat path already present in the observation.  An
        # absolute distance reward pays the policy forever for standing still;
        # positive arc progress pays only forward motion in the commanded direction.
        arc = self._arc_root[base_rows]
        clear_score = (
            torch.clamp(arc / max(self._ss_clear_arc_dist, 1e-6), 0.0, 1.0)
            if self._ss_clear_arc_dist > 0.0
            else torch.clamp(
                (root_dist - 0.40) / max(self._ss_body_clear - 0.40, 1e-6),
                0.0, 1.0,
            )
        )
        prev_arc = self._prev_arc[base_rows]
        arc_speed = (arc - prev_arc) / self.dt
        cmd_speed = self._m_at(arc, base_rows) / 1.6
        signed_progress = (arc_speed / cmd_speed.clamp(min=1e-4)).clamp(-1.0, 1.0)
        progress = (signed_progress if self._ss_clear_signed
                    else signed_progress.clamp(0.0, 1.0))
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
        clear_motion = (
            self._ss_clear_progress_w * progress
            + self._ss_clear_speed_w * speed_match
        )
        base_disp = (base[:, 0:2] - self._ss_latched_base[:, 0:2]).norm(dim=-1)
        base_lin = base[:, 7:10].norm(dim=-1)
        base_ang = base[:, 10:13].norm(dim=-1)
        clear_base_penalty = (
            self._ss_clear_base_disp_w * torch.clamp(
                base_disp / self._ss_clear_xy_tol, 0.0, 1.0
            )
            + self._ss_clear_base_lin_w * torch.clamp(
                base_lin / self._ss_clear_stable_lin, 0.0, 1.0
            )
            + self._ss_clear_base_ang_w * torch.clamp(
                base_ang / self._ss_clear_stable_ang, 0.0, 1.0
            )
        )

        # At contact this is exactly zero.  Support/stability become valuable
        # only while both hands are currently clear; the gate is not latched.
        release_local = 0.50 * h + clear_now.float() * (
            0.30 * support + 0.20 * stable
        )
        release_r = release_local
        if self._ss_release_carry_bridge:
            released_r = 0.50 + 0.30 * support + 0.20 * stable
            # With sequential completion floors enabled, interpolate the
            # *total* RELEASE reward. At h=0 this exactly preserves the
            # preceding native CARRY reward; at h=1 it reaches the same
            # carry_done + RELEASE-local ceiling used by the unbridged path.
            # This removes the CARRY -> RELEASE cliff without changing the
            # RELEASE -> CLEAR floor or any observation/model shape.
            if self._ss_sequential_reward_mask:
                released_r = self._ss_carry_done_r + released_r
            release_r = (1.0 - h) * carry_r + h * released_r
        release_phase = self._ss_phase == self.RELEASE
        release_delta_h = h - self._ss_prev_release_h
        release_progress = self._ss_release_progress_w * release_delta_h
        release_after_grace = (
            self._ss_release_age >= self._ss_release_hold_grace_steps
        ).float()
        release_hold_penalty = (
            self._ss_release_hold_pen_w
            * (1.0 - h)
            * release_after_grace
        )
        release_r = release_r + release_progress - release_hold_penalty
        self._ss_prev_release_h = torch.where(
            release_phase, h, self._ss_prev_release_h
        )

        def tb_release(value):
            return torch.where(release_phase, value, nan)

        release_hand_positive = 0.50 * h
        release_support_positive = clear_now.float() * (
            0.30 * support + 0.20 * stable
        )
        release_transition_bonus = (
            self._ss_bonus_pending.float() * self._ss_transition_bonus
        )
        release_floor = (
            self._ss_carry_done_r
            if self._ss_sequential_reward_mask
            and not self._ss_release_carry_bridge
            else 0.0
        )
        release_total = (
            release_r
            + release_floor
            + release_transition_bonus
            - self._ss_foot_box_w * foot_penalty
        )
        self.extras["tb/stack_reward/release_total"] = tb_release(release_total)
        self.extras["tb/stack_reward/release_local"] = tb_release(release_local)
        self.extras["tb/stack_reward/release_hand_positive"] = tb_release(
            release_hand_positive
        )
        self.extras["tb/stack_reward/release_support_positive"] = tb_release(
            release_support_positive
        )
        self.extras["tb/stack_reward/release_foot_penalty"] = tb_release(
            -self._ss_foot_box_w * foot_penalty
        )
        self.extras["tb/stack_reward/release_transition_bonus"] = tb_release(
            release_transition_bonus
        )
        self.extras["tb/stack_reward/release_progress"] = tb_release(
            release_progress
        )
        self.extras["tb/stack_reward/release_hold_penalty"] = tb_release(
            -release_hold_penalty
        )
        self.extras["tb/stack_state/release_hand_factor"] = tb_release(h)
        self.extras["tb/stack_state/release_hand_clear_rate"] = tb_release(
            clear_now.float()
        )
        self.extras["tb/stack_state/release_streak_fraction"] = tb_release(
            torch.clamp(
                self._ss_hand_count.float() / max(self._ss_hand_steps, 1),
                0.0,
                1.0,
            )
        )
        self.extras["tb/stack_state/release_support"] = tb_release(support)
        self.extras["tb/stack_state/release_stable"] = tb_release(stable)
        self.extras["tb/stack_state/release_foot_distance"] = tb_release(
            foot_dist
        )
        self.extras["tb/stack_state/release_age"] = tb_release(
            self._ss_release_age.float()
        )
        static_gate = clear_motion if self._ss_clear_motion_gate else 1.0
        ms20_clear_r = clear_now.float() * (
            static_gate * (0.40 * support + 0.25 * stable) + 0.35 * clear_score
        )
        stack_r = clear_now.float() * (0.40 * support + 0.25 * stable) + 0.35 * clear_score
        positive_steer = clear_now.float() * path_quality * clear_motion
        clear_r = ms20_clear_r + self._ss_clear_steer_w * positive_steer - clear_base_penalty
        if self._ss_negative_clear_reward:
            # arc is a forward-only path ratchet, so it cannot distinguish
            # standing still from moving in the wrong direction. Use the actual
            # root velocity projected onto the commanded retreat direction.
            roots = self.humanoid_rows(self._humanoid_root_states)[base_rows]
            v_along = (roots[:, 7:9] * self._ss_retreat_dir).sum(dim=-1)
            speed_denom = cmd_speed.clamp(min=1e-4)
            move_ratio = torch.clamp(v_along / speed_denom, 0.0, 1.0)
            reverse_ratio = torch.clamp(-v_along / speed_denom, 0.0, 1.0)
            min_speed = self._ss_clear_move_min_frac * cmd_speed
            stall_ratio = torch.clamp(
                (min_speed - v_along) / min_speed.clamp(min=1e-4),
                0.0,
                1.0,
            )
            after_grace = (
                self._ss_clear_age >= self._ss_clear_grace_steps
            ).float()
            # CLEAR is entered only after both hands were clear.  Any later
            # recontact removes all positive movement credit and incurs the full
            # recontact penalty, instead of leaving a near-threshold loophole.
            move_reward = (
                self._ss_clear_move_w
                * clear_now.float()
                * path_quality
                * move_ratio
            )
            hand_penalty = (
                self._ss_clear_hand_pen_w * (~clear_now).float()
            )
            stall_penalty = self._ss_clear_stall_pen_w * stall_ratio * after_grace
            reverse_penalty = (
                self._ss_clear_reverse_pen_w * reverse_ratio * after_grace
            )
            clear_r = (
                move_reward
                - hand_penalty
                - stall_penalty
                - reverse_penalty
                - clear_base_penalty
            )

            clear_phase = self._ss_phase == self.CLEAR
            nan = torch.full_like(clear_r, float("nan"))

            def tb_clear(value):
                return torch.where(clear_phase, value, nan)

            # Signed contributions and raw state rates are logged separately so
            # TensorBoard shows whether CLEAR fails at hand clearance, movement,
            # direction, box stability, or foot clearance.
            transition_bonus = (
                self._ss_bonus_pending.float() * self._ss_transition_bonus
            )
            self.extras["tb/stack_reward/clear_total"] = tb_clear(
                clear_r
                - self._ss_foot_box_w * foot_penalty
                + transition_bonus
            )
            self.extras["tb/stack_reward/move_positive"] = tb_clear(move_reward)
            self.extras["tb/stack_reward/hand_penalty"] = tb_clear(-hand_penalty)
            self.extras["tb/stack_reward/stall_penalty"] = tb_clear(-stall_penalty)
            self.extras["tb/stack_reward/reverse_penalty"] = tb_clear(-reverse_penalty)
            self.extras["tb/stack_reward/base_penalty"] = tb_clear(-clear_base_penalty)
            self.extras["tb/stack_reward/foot_penalty"] = tb_clear(
                -self._ss_foot_box_w * foot_penalty
            )
            self.extras["tb/stack_reward/transition_bonus"] = tb_clear(
                transition_bonus
            )
            self.extras["tb/stack_state/hand_factor"] = tb_clear(h)
            self.extras["tb/stack_state/hand_clear_rate"] = tb_clear(clear_now.float())
            self.extras["tb/stack_state/recontact_rate"] = tb_clear((~clear_now).float())
            self.extras["tb/stack_state/move_ratio"] = tb_clear(move_ratio)
            self.extras["tb/stack_state/move_ok_rate"] = tb_clear(
                (v_along >= min_speed).float()
            )
            self.extras["tb/stack_state/stall_rate"] = tb_clear(
                (v_along < min_speed).float()
            )
            self.extras["tb/stack_state/reverse_rate"] = tb_clear(
                (v_along < 0.0).float()
            )
            self.extras["tb/stack_state/v_along"] = tb_clear(v_along)
            self.extras["tb/stack_state/v_command"] = tb_clear(cmd_speed)
            self.extras["tb/stack_state/retreat_arc"] = tb_clear(arc)
        if self._ss_sequential_reward_mask:
            if not self._ss_release_carry_bridge:
                release_r = self._ss_carry_done_r + release_r
            clear_r = self._ss_carry_done_r + self._ss_release_done_r + clear_r
            stack_r = (
                self._ss_carry_done_r
                + self._ss_release_done_r
                + self._ss_clear_done_r
                + stack_r
            )
        replacement = torch.where(self._ss_phase == self.RELEASE, release_r, clear_r)
        replacement = torch.where(self._ss_phase >= self.STACK, stack_r, replacement)
        replacement = torch.where(
            self._ss_phase == self.SUCCESS, replacement + 0.5, replacement
        )
        success_bonus = (
            self._ss_success_bonus_pending.float() * self._ss_success_bonus
        )
        self._ss_dbg_success_bonus_sum += success_bonus
        replacement = replacement + success_bonus
        self.extras["tb/stack_reward/success_bonus"] = success_bonus
        replacement = replacement + self._ss_bonus_pending.float() * self._ss_transition_bonus
        replacement = (
            replacement
            + self._ss_clear_bonus_pending.float() * self._ss_clear_bonus
        )
        foot_active = (self._ss_phase >= self.RELEASE) & (self._ss_phase <= self.SUCCESS)
        replacement = replacement - (
            self._ss_foot_box_w * foot_penalty * foot_active.float()
        )
        success_pending = self._ss_success_bonus_pending
        self.rew_buf[top_rows[success_pending]] += success_bonus[success_pending]
        self._ss_bonus_pending[:] = False
        self._ss_clear_bonus_pending[:] = False
        self._ss_success_bonus_pending[:] = False
        self._ss_above_bonus_pending[:] = False
        self.rew_buf[base_rows[post]] = replacement[post]

        self.extras["stack_clear_arc_speed"] = arc_speed
        self.extras["stack_clear_cmd_speed"] = cmd_speed
        self.extras["stack_clear_path_quality"] = path_quality
        self.extras["stack_clear_positive_steer"] = positive_steer
        self.extras["stack_clear_base_penalty"] = clear_base_penalty

        if bool(failed.any()):
            self.rew_buf[base_rows[failed]] = -1.0
        return

    # --------------------------------------------------------------- controller

    def _hold_steer(self, rows):
        """Stop a completed controller path without leaving a stale direction."""
        if len(rows) == 0:
            return
        roots = self.humanoid_rows(self._humanoid_root_states)[rows, 0:2]
        self._gt_path[rows] = roots[:, None, :]
        self._s_end[rows] = 0.0
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = 0.0
        self._prev_arc[rows] = 0.0
        self._mscale[rows] = 0.0

    def _set_retreat_path(self, env_ids, base_rows, base):
        roots = self.humanoid_rows(self._humanoid_root_states)[base_rows, 0:2]
        away_raw = roots - base[:, 0:2]
        norm = away_raw.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(away_raw)
        fallback[:, 0] = 1.0
        away = torch.where(
            norm > 1e-4, away_raw / norm.clamp(min=1e-4), fallback
        )
        left = torch.stack((-away[:, 1], away[:, 0]), dim=-1)
        top_side = self._ss_stage_goal[env_ids, 0:2] - base[:, 0:2]
        use_left = (left * top_side).sum(dim=-1, keepdim=True) <= 0.0
        side = torch.where(use_left, left, -left)

        if self._ss_clear_route_around:
            # Move tangentially around the placed box, on the side opposite the
            # waiting top carrier.  Keeping the radial coordinate outside the
            # expanded footprint avoids commanding a blind backward walk.
            half_diag = 0.5 * self._box_lib._box_size[base_rows, 0:2].norm(dim=-1)
            radial_clearance = torch.maximum(
                norm.squeeze(-1), half_diag + self._ss_clear_route_margin
            )
            corner = min(0.35, 0.5 * self._ss_retreat_dist)
            waypoint = (
                base[:, 0:2]
                + away * radial_clearance[:, None]
                + side * corner
            )
            clear = (
                base[:, 0:2]
                + away * radial_clearance[:, None]
                + side * self._ss_retreat_dist
            )
            move_dir = torch.nn.functional.normalize(clear - roots, dim=-1)
            lat_max = 0.0
        else:
            move_dir = away
            if self._ss_retreat_side_deg > 0.0:
                angle = math.radians(self._ss_retreat_side_deg)
                move_dir = math.cos(angle) * away + math.sin(angle) * side
            clear = base[:, 0:2] + move_dir * self._ss_retreat_dist
            waypoint = roots + move_dir * 0.15
            lat_max = 0.35

        self._ss_retreat_dir[env_ids] = move_dir
        self._ss_retreat_goal[env_ids, 0:2] = clear
        self._ss_retreat_goal[env_ids, 2] = base[:, 2]
        self._steer_tick += 1
        path, s_waypoint, n_end = sp.gen_full_v2(
            roots, waypoint, clear, self.steer_seed + self._steer_tick + int(env_ids[0]),
            0.0, lat_max, 120.0, p_two=0.0, skew=0.8,
            spread=(0.85, 1.8), lat_frac=0.25, with_end=True,
        )
        self._gt_path[base_rows] = path
        self._s_end[base_rows] = (n_end.float() - 1.0) * sp.DS
        self._arc_root[base_rows] = 0.0
        self._arc_box[base_rows] = s_waypoint
        self._prev_arc[base_rows] = 0.0
        self._mscale[base_rows] = self._ss_retreat_scale
    def _virtual_retreat_carry_obs(self, rows, env_ids):
        """Replace the real placed-box observation with a virtual CLEAR box."""
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        rigid = self.humanoid_rows(self._rigid_body_pos)[rows]
        real_box = self.humanoid_rows(self._box_states)[rows]
        heading_inv = torch_utils.calc_heading_quat_inv(roots[:, 3:7])

        if self._ss_virtual_rear_box:
            # The physical placed box remains in the simulator so the stack
            # and foot-contact penalty stay valid. Only the carry-token input
            # sees a stationary phantom box on the rear retreat endpoint.
            box_pos = self._ss_retreat_goal[env_ids].clone()
            box_pos[:, 2] = real_box[:, 2]
            box_vel = torch.zeros_like(roots[:, 7:10])
        else:
            box_pos = rigid[:, self._key_body_ids[[0, 1]]].mean(dim=1)
            box_vel = roots[:, 7:10]
        box_rot = real_box[:, 3:7]
        box_ang_vel = torch.zeros_like(box_vel)

        local_box_vel = quat_rotate(heading_inv, box_vel)
        local_box_ang_vel = quat_rotate(heading_inv, box_ang_vel)
        local_box_pos = quat_rotate(heading_inv, box_pos - roots[:, 0:3])
        local_box_rot = torch_utils.quat_to_tan_norm(
            quat_mul(heading_inv, box_rot)
        )

        bps = self._box_lib._box_bps[rows]
        n = bps.shape[1]
        box_rot_exp = box_rot[:, None, :].expand(-1, n, -1)
        world_bps = (
            quat_rotate(
                box_rot_exp.reshape(-1, 4), bps.reshape(-1, 3)
            ).view(len(rows), n, 3)
            + box_pos[:, None, :]
        )
        heading_exp = heading_inv[:, None, :].expand(-1, n, -1)
        root_exp = roots[:, None, 0:3].expand(-1, n, -1)
        local_bps = quat_rotate(
            heading_exp.reshape(-1, 4),
            (world_bps - root_exp).reshape(-1, 3),
        ).view(len(rows), -1)

        carry_goal = self._ss_retreat_goal[env_ids].clone()
        carry_goal[:, 2] = box_pos[:, 2]
        local_goal = quat_rotate(heading_inv, carry_goal - roots[:, 0:3])
        return torch.cat(
            (
                local_box_vel,
                local_box_ang_vel,
                local_box_pos,
                local_box_rot,
                local_bps,
                local_goal,
            ),
            dim=-1,
        )

    def _compute_task_obs(self, env_ids=None):
        obs = super()._compute_task_obs(env_ids)
        if not hasattr(self, "_ss_phase") or not (
            self._ss_virtual_retreat
            or self._ss_zero_carry_obs
            or self._ss_dynamic_carry_mask
        ):
            return obs

        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        agent = rows % self.num_agents
        retreat = (
            (~self._ss_rehearsal[env])
            & (agent == self._ss_base_agent[env])
            & (self._ss_phase[env] >= self.CLEAR)
            & (self._ss_phase[env] <= self.SUCCESS)
        )
        if not bool(retreat.any()):
            return obs

        out = obs.clone()
        carry_start = TEAMMATE_DIM * (self.num_agents - 1) + self.steer_dim()
        carry_dim = CARRY_HI - CARRY_LO
        if self._ss_zero_carry_obs or self._ss_dynamic_carry_mask:
            # Zero observation and attention masking are deliberately separate.
            # STACK_ZERO_CARRY_OBS leaves both carry token positions active in
            # the frozen Transformer; STACK_DYNAMIC_CARRY_MASK additionally
            # masks those positions in the network.
            out[retreat, carry_start:carry_start + 2 * carry_dim] = 0.0
        else:
            virtual = self._virtual_retreat_carry_obs(rows[retreat], env[retreat])
            out[retreat, carry_start:carry_start + carry_dim] = virtual
            out[retreat, carry_start + carry_dim:carry_start + 2 * carry_dim] = virtual
        return out

    def _update_stack_controller(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        (base_rows, top_rows, base, top, hand_dist, xy_err, z_err,
         _, _, root_dist) = self._base_features()

        stack_env = ~self._ss_rehearsal
        carry = (self._ss_phase == self.CARRY) & stack_env
        clear_now = hand_dist >= self._ss_hand_clear
        foot_dist = (
            self._all_foot_surface_distance(base_rows, top_rows)
            if self._ss_carry_foot_gate
            else self._foot_surface_distance(base_rows)
        )
        foot_clear_now = foot_dist >= self._ss_foot_clear
        entry_foot_ok = (
            foot_clear_now
            if self._ss_entry_foot_gate
            else torch.ones_like(foot_clear_now)
        )
        release_foot_ok = (
            foot_clear_now
            if self._ss_release_foot_gate
            else torch.ones_like(foot_clear_now)
        )
        entry_xy = xy_err <= self._ss_xy_tol
        entry_z = z_err <= self._ss_z_tol
        entry_lin = base[:, 7:10].norm(dim=-1) <= self._ss_entry_lin
        entry_ang = base[:, 10:13].norm(dim=-1) <= self._ss_entry_ang
        entry_upright = self._upright_score(base[:, 3:7]) >= math.cos(
            math.radians(10.0)
        )
        stable_placeable = (
            entry_xy & entry_z & entry_lin & entry_ang & entry_upright
        )
        delivered = self._ep_finish[base_rows] >= 0
        placeable = delivered if self._ss_entry_delivered else stable_placeable
        if self._ss_entry_lowered:
            placeable = placeable & entry_z
        picked = self._ss_dbg_pick_step[base_rows] >= 0
        entry_joint = (
            picked & clear_now
            if self._ss_hand_only_switch
            else placeable & entry_foot_ok
        )
        diag = self._ss_dbg_gate_counts
        diag[:, self.DIAG_CARRY_STEPS] += carry.float()
        for column, condition in (
            (self.DIAG_ENTRY_XY, entry_xy),
            (self.DIAG_ENTRY_Z, entry_z),
            (self.DIAG_ENTRY_LIN, entry_lin),
            (self.DIAG_ENTRY_ANG, entry_ang),
            (self.DIAG_ENTRY_UPRIGHT, entry_upright),
            (self.DIAG_ENTRY_FOOT, foot_clear_now),
            (self.DIAG_ENTRY_PLACEABLE, placeable),
            (self.DIAG_ENTRY_JOINT, entry_joint),
        ):
            diag[:, column] += (carry & condition).float()
        self._ss_entry_count = torch.where(
            carry & placeable & entry_foot_ok,
            self._ss_entry_count + 1,
            torch.where(carry, torch.zeros_like(self._ss_entry_count), self._ss_entry_count),
        )
        diag[:, self.DIAG_ENTRY_STREAK_MAX] = torch.maximum(
            diag[:, self.DIAG_ENTRY_STREAK_MAX], self._ss_entry_count.float()
        )
        if self._ss_hand_only_switch:
            enter_release = carry & picked & clear_now
        else:
            enter_release = carry & (self._ss_entry_count >= self._ss_entry_steps)
        if bool(enter_release.any()):
            ids = torch.nonzero(enter_release, as_tuple=False).squeeze(-1)
            rows = base_rows[ids]
            self._ss_dbg_place_step[rows] = self.progress_rows()[rows]
            self._ss_phase[ids] = self.RELEASE
            self._ss_latched_base[ids] = base[ids, 0:3]
            self._ss_release_age[ids] = 0
            self._ss_prev_release_h[ids] = torch.clamp(
                hand_dist[ids] / self._ss_hand_clear, 0.0, 1.0
            )
            # Existing steering token only: zero window means stop the object.
            self._mscale[rows] = 0.0
            self._compute_observations(ids)

        # The top carrier must arrive, keep both hands near the box, and settle
        # before the staging latch is valid.  A broken grasp unlatches staging so
        # CLEAR cannot transition into an unreachable STACK state.
        stage_dist = (top[:, 0:3] - self._ss_stage_goal).norm(dim=-1)
        top_picked = self._ss_dbg_pick_step[top_rows] >= 0
        top_grasped = top_picked & (
            self._hand_surface_distances(top_rows).amax(dim=-1)
            <= self._ss_stage_hand_tol
        )
        top_stage_stable = (
            (top[:, 7:10].norm(dim=-1) <= self._ss_stage_stable_lin)
            & (top[:, 10:13].norm(dim=-1) <= self._ss_stage_stable_ang)
        )
        pre_stack = stack_env & (self._ss_phase < self.STACK)
        if self._ss_stage_hold_steps > 0:
            stage_candidate = (
                pre_stack
                & top_grasped
                & top_stage_stable
                & (stage_dist <= self._ss_stage_tol)
            )
            self._ss_stage_count = torch.where(
                stage_candidate,
                self._ss_stage_count + 1,
                torch.where(
                    pre_stack,
                    torch.zeros_like(self._ss_stage_count),
                    self._ss_stage_count,
                ),
            )
            newly_staged = (
                pre_stack
                & (~self._ss_staged)
                & (self._ss_stage_count >= self._ss_stage_hold_steps)
            )
            stage_valid = top_grasped & (stage_dist <= 1.5 * self._ss_stage_tol)
            lost_stage = pre_stack & self._ss_staged & (~stage_valid)
            if bool(lost_stage.any()):
                ids = torch.nonzero(lost_stage, as_tuple=False).squeeze(-1)
                self._ss_staged[ids] = False
                self._ss_stage_count[ids] = 0
                self._mscale[top_rows[ids]] = 1.0
                self._compute_observations(ids)
        else:
            newly_staged = (
                pre_stack & (~self._ss_staged) & (stage_dist <= self._ss_stage_tol)
            )
        if bool(newly_staged.any()):
            ids = torch.nonzero(newly_staged, as_tuple=False).squeeze(-1)
            self._ss_staged[ids] = True
            self._ss_dbg_staged_step[ids] = self.progress_buf[ids]
            self._mscale[top_rows[ids]] = 0.0
            self._compute_observations(ids)

        release = self._ss_phase == self.RELEASE
        self._ss_release_age = torch.where(
            release,
            self._ss_release_age + 1,
            torch.zeros_like(self._ss_release_age),
        )
        release_joint = clear_now & release_foot_ok
        diag[:, self.DIAG_RELEASE_STEPS] += release.float()
        diag[:, self.DIAG_RELEASE_HAND] += (release & clear_now).float()
        diag[:, self.DIAG_RELEASE_FOOT] += (release & foot_clear_now).float()
        diag[:, self.DIAG_RELEASE_JOINT] += (release & release_joint).float()
        self._ss_hand_count = torch.where(
            release & release_joint,
            self._ss_hand_count + 1,
            torch.where(release, torch.zeros_like(self._ss_hand_count), self._ss_hand_count),
        )
        diag[:, self.DIAG_RELEASE_STREAK_MAX] = torch.maximum(
            diag[:, self.DIAG_RELEASE_STREAK_MAX], self._ss_hand_count.float()
        )
        just_released = release & (self._ss_hand_count >= self._ss_hand_steps)
        if bool(just_released.any()):
            ids = torch.nonzero(just_released, as_tuple=False).squeeze(-1)
            rows = base_rows[ids]
            self._ss_dbg_release_step[rows] = self.progress_rows()[rows]
            self._ss_phase[ids] = self.CLEAR
            self._ss_clear_age[ids] = 0
            self._ss_bonus_pending[ids] = True
            self._set_retreat_path(ids, rows, base[ids])
            self._compute_observations(ids)

        clear = self._ss_phase == self.CLEAR
        base_still_supported = (
            (xy_err <= self._ss_clear_xy_tol)
            & (z_err <= self._ss_clear_z_tol)
            & (base[:, 7:10].norm(dim=-1) <= self._ss_clear_stable_lin)
            & (base[:, 10:13].norm(dim=-1) <= self._ss_clear_stable_ang)
        )
        retreat_done = (
            self._arc_root[base_rows] >= self._ss_clear_arc_dist
            if self._ss_clear_arc_dist > 0.0
            else root_dist >= self._ss_body_clear
        )
        ready_to_stack = clear & clear_now & retreat_done
        if self._ss_clear_hard_gate:
            ready_to_stack &= base_still_supported
        if self._ss_require_staged:
            ready_to_stack &= (
                self._ss_staged
                & top_grasped
                & (stage_dist <= 1.5 * self._ss_stage_tol)
            )
        diag[:, self.DIAG_CLEAR_STEPS] += clear.float()
        diag[:, self.DIAG_CLEAR_HAND] += (clear & clear_now).float()
        diag[:, self.DIAG_CLEAR_BODY] += (
            clear & (root_dist >= self._ss_body_clear)
        ).float()
        diag[:, self.DIAG_CLEAR_STABLE] += (clear & base_still_supported).float()
        diag[:, self.DIAG_CLEAR_JOINT] += ready_to_stack.float()
        diag[:, self.DIAG_CLEAR_FOOT] += (clear & foot_clear_now).float()
        diag[:, self.DIAG_RETREAT_ROOT_DIST_MAX] = torch.where(
            clear,
            torch.maximum(diag[:, self.DIAG_RETREAT_ROOT_DIST_MAX], root_dist),
            diag[:, self.DIAG_RETREAT_ROOT_DIST_MAX],
        )
        diag[:, self.DIAG_RETREAT_ARC_MAX] = torch.where(
            clear,
            torch.maximum(diag[:, self.DIAG_RETREAT_ARC_MAX], self._arc_root[base_rows]),
            diag[:, self.DIAG_RETREAT_ARC_MAX],
        )
        if bool(ready_to_stack.any()):
            ids = torch.nonzero(ready_to_stack, as_tuple=False).squeeze(-1)
            rows = top_rows[ids]
            base_event_rows = base_rows[ids]
            self._ss_dbg_clear_step[base_event_rows] = self.progress_rows()[base_event_rows]
            self._ss_phase[ids] = self.STACK
            self._ss_clear_bonus_pending[ids] = True
            self._ss_above_seen[ids] = False
            self._ss_above_bonus_pending[ids] = False
            self._ss_top_release_age[ids] = 0
            self._ss_stack_age[ids] = 0
            self._ss_top_place_count[ids] = 0
            self._ss_top_settled[ids] = False
            self._ss_prev_top_h[ids] = 0.0
            self._ss_prev_top_dist[ids] = 0.0
            if self._ss_clear_stop_on_stack:
                self._hold_steer(base_event_rows)
            self._box_tar_pos[rows, 0:2] = base[ids, 0:2]
            base_size = self._box_lib._box_size[base_rows[ids]]
            top_size = self._box_lib._box_size[rows]
            self._box_tar_pos[rows, 2] = base[ids, 2] + 0.5 * (
                base_size[:, 2] + top_size[:, 2]
            )
            start_dist = (
                top[ids, 0:3] - self._box_tar_pos[rows]
            ).norm(dim=-1)
            self._ss_prev_top_dist[ids] = start_dist
            self._ss_dbg_stack_dist_start[ids] = start_dist
            self._ss_dbg_stack_dist_min[ids] = start_dist
            self._reset_steer(rows)
            self._mscale[rows] = self._ss_top_scale
            self._compute_observations(ids)

        self._ss_clear_age = torch.where(
            self._ss_phase == self.CLEAR,
            self._ss_clear_age + 1,
            torch.zeros_like(self._ss_clear_age),
        )

        active = self._ss_phase == self.STACK
        if bool(active.any()):
            ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            rows = top_rows[ids]
            desired = self._box_tar_pos[rows].clone()
            desired[:, 0:2] = base[ids, 0:2]
            base_size = self._box_lib._box_size[base_rows[ids]]
            top_size = self._box_lib._box_size[rows]
            desired[:, 2] = base[ids, 2] + 0.5 * (base_size[:, 2] + top_size[:, 2])
            self._box_tar_pos[rows] = desired
            target = self._box_tar_pos[rows]
            top_xy_err = (top[ids, 0:2] - target[:, 0:2]).norm(dim=-1)
            top_z_err = (top[ids, 2] - target[:, 2]).abs()
            top_dist = torch.sqrt(top_xy_err.square() + top_z_err.square())
            above_now = (
                (top_xy_err <= self._ss_above_xy_tol)
                & (top_z_err <= self._ss_above_z_tol)
            )
            base_shift_ok = (
                (base[ids, 0:2] - self._ss_base_goal[ids, 0:2]).norm(dim=-1)
                <= self._ss_drop_xy
            )
            top_place_ok = (
                (top_xy_err <= self._ss_top_xy_tol)
                & (top_z_err <= self._ss_top_z_tol)
                & (top[ids, 7:10].norm(dim=-1) <= self._ss_top_stable_lin)
                & (top[ids, 10:13].norm(dim=-1) <= self._ss_top_stable_ang)
                & (
                    self._upright_score(top[ids, 3:7])
                    >= math.cos(math.radians(self._ss_top_upright_deg))
                )
                & base_shift_ok
            )
            self._ss_stack_age[ids] += 1
            self._ss_dbg_stack_steps[ids] += 1
            self._ss_dbg_stack_dist_min[ids] = torch.minimum(
                self._ss_dbg_stack_dist_min[ids], top_dist
            )
            top_grasp_now = (
                self._hand_surface_distances(rows).amax(dim=-1)
                <= self._ss_stage_hand_tol
            )
            self._ss_dbg_stack_grasp_steps[ids] += top_grasp_now.long()
            self._ss_dbg_stack_place_steps[ids] += top_place_ok.long()
            if self._ss_top_settle_steps > 0:
                self._ss_top_place_count[ids] = torch.where(
                    top_place_ok,
                    self._ss_top_place_count[ids] + 1,
                    torch.zeros_like(self._ss_top_place_count[ids]),
                )
                settled_now = (
                    self._ss_top_place_count[ids] >= self._ss_top_settle_steps
                )
            else:
                settled_now = above_now
            first_settled = settled_now & (self._ss_dbg_settled_step[ids] < 0)
            if bool(first_settled.any()):
                settled_ids = ids[first_settled]
                self._ss_dbg_settled_step[settled_ids] = self.progress_buf[settled_ids]
            self._ss_top_settled[ids] = settled_now
            first_above = above_now & (~self._ss_above_seen[ids])
            if bool(first_above.any()):
                above_ids = ids[first_above]
                self._ss_dbg_above_step[above_ids] = self.progress_buf[above_ids]
            self._ss_above_bonus_pending[ids] |= first_above
            self._ss_above_seen[ids] |= above_now
            self._ss_top_release_age[ids] = torch.where(
                settled_now,
                self._ss_top_release_age[ids] + 1,
                torch.zeros_like(self._ss_top_release_age[ids]),
            )
            top_hand_dist = self._hand_surface_distance(rows)
            top_hand_ok = (
                top_hand_dist >= self._ss_top_hand_clear
                if self._ss_top_require_hand_clear
                else torch.ones_like(above_now)
            )
            top_ok = top_place_ok & top_hand_ok
            self._ss_top_count[ids] = torch.where(
                top_ok,
                self._ss_top_count[ids] + 1,
                torch.zeros_like(self._ss_top_count[ids]),
            )
            success = ids[self._ss_top_count[ids] >= self._ss_top_steps]
            if len(success) > 0:
                success_base_rows = base_rows[success]
                self._ss_dbg_success_step[success_base_rows] = self.progress_rows()[
                    success_base_rows
                ]
                self._ss_phase[success] = self.SUCCESS
                self._ss_success[success] = True
                self._ss_success_bonus_pending[success] = True
                self._hold_steer(top_rows[success])
            self._compute_observations(ids)

        # No re-grasp controller: after release, losing the base ends the trial.
        post = (self._ss_phase >= self.RELEASE) & (self._ss_phase < self.SUCCESS)
        base_shift = (base[:, 0:2] - self._ss_base_goal[:, 0:2]).norm(dim=-1)
        dropped = post & ((base_shift > self._ss_drop_xy) | (base[:, 2] < -0.05))
        if bool(dropped.any()):
            ids = torch.nonzero(dropped, as_tuple=False).squeeze(-1)
            rows = base_rows[ids]
            self._ss_dbg_fail_step[rows] = self.progress_rows()[rows]
            self._ss_phase[ids] = self.FAILED
            self._ss_failed[ids] = True
            self._ss_dbg_end_reason[ids] = 6
            self.reset_buf[ids] = 1

        self.extras["stack_phase"] = self._ss_phase.float()
        self.extras["stack_base_agent"] = self._ss_base_agent.float()
        self.extras["stack_hand_distance"] = hand_dist
        self.extras["stack_release_done"] = (self._ss_phase >= self.CLEAR).float()
        self.extras["stack_clear_done"] = (self._ss_phase >= self.STACK).float()
        self.extras["stack_retreat_done"] = retreat_done.float()
        self.extras["stack_staged"] = self._ss_staged.float()
        self.extras["stack_success"] = self._ss_success.float()
        self.extras["stack_failed"] = self._ss_failed.float()
        self.extras["tb/stack_phase/release_fraction"] = (
            self._ss_phase == self.RELEASE
        ).float()
        self.extras["tb/stack_phase/clear_fraction"] = (
            self._ss_phase == self.CLEAR
        ).float()
        self.extras["tb/stack_phase/stack_fraction"] = (
            self._ss_phase == self.STACK
        ).float()
        self.extras["tb/stack_phase/failed_fraction"] = self._ss_failed.float()
        return

    def _compute_reset(self):
        super()._compute_reset()
        if self._ss_phase_steps <= 0:
            return

        # The generated cfg gives this task PRE+STACK total room. Preserve all
        # true terminations and non-timeout completion signals from the parent,
        # but cap pre-STACK work at PRE and grant a fresh STACK_PHASE budget.
        early = self._terminate_buf.bool()
        global_timeout = self.progress_buf >= self.max_episode_length - 1
        parent_other = self.reset_buf.bool() & (~early) & (~global_timeout)
        pre_timeout = (
            (self._ss_phase < self.STACK)
            & (self.progress_buf >= self._ss_pre_steps - 1)
        )
        stack_timeout = (
            (self._ss_phase == self.STACK)
            & (self._ss_stack_age >= self._ss_phase_steps)
        )
        reset = early | parent_other | pre_timeout | stack_timeout | global_timeout
        self.reset_buf[:] = reset.long()

        unset = self._ss_dbg_end_reason < 0
        reason = torch.full_like(self._ss_dbg_end_reason, -1)
        reason = torch.where(global_timeout, torch.full_like(reason, 4), reason)
        reason = torch.where(parent_other, torch.full_like(reason, 5), reason)
        reason = torch.where(stack_timeout, torch.full_like(reason, 3), reason)
        reason = torch.where(pre_timeout, torch.full_like(reason, 2), reason)
        reason = torch.where(early, torch.ones_like(reason), reason)
        record = reset & unset
        self._ss_dbg_end_reason[record] = reason[record]
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._update_stage_metrics()
        self._update_stack_controller()
        return
