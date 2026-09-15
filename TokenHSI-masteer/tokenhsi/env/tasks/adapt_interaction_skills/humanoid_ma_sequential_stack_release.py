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
from env.tasks.multi_task.humanoid_traj_sit_carry_climb import (
    compute_carry_reward,
    compute_handheld_reward,
    compute_putdown_reward,
    compute_walk_reward,
)
from tokenhsi.utils import steer_path as sp
from tokenhsi.utils import stack_bootstrap
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

        self._ss_bootstrap_import = None
        bootstrap_path = os.environ.get("STACK_BOOTSTRAP_LOAD", "")
        if bootstrap_path:
            self._ss_bootstrap_import = stack_bootstrap.select_bank(
                stack_bootstrap.read_bank(bootstrap_path),
                _i("STACK_BOOTSTRAP_INDEX", 0), cfg["env"]["numEnvs"],
            )
        self._ss_bootstrap_save = os.environ.get("STACK_BOOTSTRAP_SAVE", "")
        self._ss_bootstrap_save_once = bool(_i("STACK_BOOTSTRAP_SAVE_ONCE", 0))
        self._ss_bootstrap_export_done = False
        self._ss_bootstrap_collect_steps = _i("STACK_BOOTSTRAP_COLLECT_STEPS", 6000)
        self._ss_bootstrap_collect_tick = 0
        self._ss_seed = _i("STACK_SEED", int(cfg.get("seed", 0)) + 19020)
        self._ss_size_margin = _f("STACK_SIZE_MARGIN", 0.05)
        self._ss_goal_x = _f("STACK_GOAL_X", 4.0)
        self._ss_goal_y = _f("STACK_GOAL_Y", 0.0)
        self._ss_stage_dist = _f("STACK_STAGE_DIST", 2.0)
        self._ss_stage_z = _f("STACK_STAGE_Z", 0.90)
        self._ss_stage_use_hand_z = bool(_i("STACK_STAGE_USE_HAND_Z", 0))
        self._ss_stage_force_zero = bool(_i("STACK_STAGE_FORCE_ZERO", 1))
        self._ss_stage_tol = _f("STACK_STAGE_TOL", 0.45)
        self._ss_stage_hand_tol = _f("STACK_STAGE_HAND_TOL", 0.35)
        self._ss_stage_stable_lin = _f("STACK_STAGE_STABLE_LIN", 0.25)
        self._ss_stage_stable_ang = _f("STACK_STAGE_STABLE_ANG", 1.0)
        self._ss_stage_hold_steps = _i("STACK_STAGE_HOLD_STEPS", 0)
        self._ss_require_staged = bool(_i("STACK_REQUIRE_STAGED", 0))
        self._ss_top_wait_at_start = bool(_i("STACK_TOP_WAIT_AT_START", 0))
        self._ss_top_commit_goal = bool(_i("STACK_TOP_COMMIT_GOAL", 0))
        self._ss_shared_goal_carry = bool(_i("STACK_SHARED_GOAL_CARRY", 0))
        self._ss_shared_wait_dist = _f("STACK_SHARED_WAIT_DIST", 0.90)
        self._ss_shared_path_clearance = _f(
            "STACK_SHARED_PATH_CLEARANCE", 0.0
        )
        self._ss_shared_path_candidates = _i(
            "STACK_SHARED_PATH_CANDIDATES", 8
        )
        self._ss_shared_path_retries = _i("STACK_SHARED_PATH_RETRIES", 1)
        self._ss_shared_path_start_margin = _f(
            "STACK_SHARED_PATH_START_MARGIN", 0.0
        )
        self._ss_top_wait_reward_w = _f("STACK_TOP_WAIT_REWARD_W", 0.0)
        self._ss_base_hold_reward_w = _f("STACK_BASE_HOLD_REWARD_W", 0.0)
        self._ss_pre_steps = _i("STACK_PRE_STEPS", 0)
        self._ss_phase_steps = _i("STACK_PHASE_STEPS", 0)
        self._ss_body_clear = _f("STACK_BODY_CLEAR", 1.0)
        self._ss_retreat_dist = _f("STACK_RETREAT_DIST", 1.5)
        self._ss_retreat_side_deg = _f("STACK_RETREAT_SIDE_DEG", 0.0)
        self._ss_retreat_random = bool(_i("STACK_RETREAT_RANDOM", 0))
        # Keep the old implicit value as the class default so ms20 sidecars
        # remain replayable; the ms21 wrapper explicitly selects 0.5.
        self._ss_retreat_scale = _f("STACK_RETREAT_SCALE", 1.0)
        self._ss_clear_route_around = bool(_i("STACK_CLEAR_ROUTE_AROUND", 0))
        self._ss_clear_route_margin = _f("STACK_CLEAR_ROUTE_MARGIN", 0.30)
        self._ss_clear_stop_on_stack = bool(_i("STACK_CLEAR_STOP_ON_STACK", 0))
        self._ss_debug_keep_wait = bool(_i("STACK_DEBUG_KEEP_WAIT", 0))
        self._ss_debug_carry_live_on_stack = bool(
            _i("STACK_DEBUG_CARRY_LIVE_ON_STACK", 0)
        )
        self._ss_debug_top_carry_target_only = bool(
            _i(
                "STACK_TOP_CARRY_TARGET_ONLY",
                _i("STACK_DEBUG_TOP_CARRY_TARGET_ONLY", 0),
            )
        )
        self._ss_top_direct_carry_reward = bool(
            _i("STACK_TOP_DIRECT_CARRY_REWARD", 0)
        )
        self._ss_debug_require_top_balanced = bool(
            _i("STACK_DEBUG_REQUIRE_TOP_BALANCED", 0)
        )
        self._ss_bootstrap_capture_stable = bool(
            _i("STACK_BOOTSTRAP_CAPTURE_STABLE", 0)
        )
        self._ss_bootstrap_top_balance_steps = _i(
            "STACK_BOOTSTRAP_TOP_BALANCE_STEPS", 0
        )
        self._ss_bootstrap_top_root_lin = _f(
            "STACK_BOOTSTRAP_TOP_ROOT_LIN", 0.10
        )
        self._ss_bootstrap_top_root_ang = _f(
            "STACK_BOOTSTRAP_TOP_ROOT_ANG", 0.50
        )
        self._ss_bootstrap_top_upright_deg = _f(
            "STACK_BOOTSTRAP_TOP_UPRIGHT_DEG", 15.0
        )
        self._ss_bootstrap_top_box_lin = _f(
            "STACK_BOOTSTRAP_TOP_BOX_LIN", 0.15
        )
        self._ss_bootstrap_top_box_ang = _f(
            "STACK_BOOTSTRAP_TOP_BOX_ANG", 0.40
        )
        self._ss_bootstrap_top_require_grasp = bool(
            _i("STACK_BOOTSTRAP_TOP_REQUIRE_GRASP", 1)
        )
        self._ss_bootstrap_frac = _f("STACK_BOOTSTRAP_FRAC", 0.0)
        self._ss_bootstrap_keep_wait = bool(_i("STACK_BOOTSTRAP_KEEP_WAIT", 0))
        self._ss_bootstrap_eval = bool(_i("STACK_BOOTSTRAP_EVAL", 0))
        self._ss_bootstrap_loop_radius = _f("STACK_BOOTSTRAP_LOOP_RADIUS", 0.0)
        self._ss_bootstrap_loop_scale = _f("STACK_BOOTSTRAP_LOOP_SCALE", 0.25)
        self._ss_bootstrap_loop_track_w = _f(
            "STACK_BOOTSTRAP_LOOP_TRACK_W", 1.0
        )
        self._ss_humanoid_stability_w = _f(
            "STACK_HUMANOID_STABILITY_W", 0.0
        )
        self._ss_trace_path = os.environ.get("STACK_TRACE", "").strip()
        self._ss_trace_events = _i("STACK_TRACE_EVENTS", 16)
        self._ss_trace_steps = _i("STACK_TRACE_STEPS", 60)

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
        # Backward-compatible by default. New runs can require the top and
        # base box faces to be parallel before the terminal success bonus is
        # paid. Cube/square symmetry is handled modulo 90 degrees below.
        self._ss_top_parallel_deg = _f("STACK_TOP_PARALLEL_DEG", 180.0)
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
        self._ss_clear_track_pen_w = _f("STACK_CLEAR_TRACK_PEN_W", 0.0)
        self._ss_clear_reverse_pen_w = _f("STACK_CLEAR_REVERSE_PEN_W", 0.5)
        self._ss_clear_facing_w = _f("STACK_CLEAR_FACING_W", 0.0)
        self._ss_clear_move_min_frac = _f("STACK_CLEAR_MOVE_MIN_FRAC", 0.20)
        self._ss_clear_grace_steps = _i("STACK_CLEAR_GRACE_STEPS", 0)
        self._ss_stop_decel_dist = _f("STACK_STOP_DECEL_DIST", 0.30)
        self._ss_stop_hold_steps = _i("STACK_STOP_HOLD_STEPS", 0)
        self._ss_stop_lin = _f("STACK_STOP_LIN", 0.10)
        self._ss_stop_ang = _f("STACK_STOP_ANG", 0.50)
        self._ss_stop_upright_deg = _f("STACK_STOP_UPRIGHT_DEG", 15.0)
        self._ss_stop_contact_force = _f("STACK_STOP_CONTACT_FORCE", 1.0)
        self._ss_stop_reward_w = _f("STACK_STOP_REWARD_W", 0.0)
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
        if self._ss_bootstrap_frac < 0.0 or self._ss_bootstrap_frac > 1.0:
            raise ValueError("STACK_BOOTSTRAP_FRAC must be in [0, 1]")
        if self._ss_bootstrap_loop_radius < 0.0:
            raise ValueError("STACK_BOOTSTRAP_LOOP_RADIUS must be non-negative")
        if self._ss_bootstrap_loop_scale <= 0.0:
            raise ValueError("STACK_BOOTSTRAP_LOOP_SCALE must be positive")
        if self._ss_bootstrap_top_balance_steps < 0:
            raise ValueError(
                "STACK_BOOTSTRAP_TOP_BALANCE_STEPS must be non-negative"
            )
        if min(
            self._ss_bootstrap_top_root_lin,
            self._ss_bootstrap_top_root_ang,
            self._ss_bootstrap_top_box_lin,
            self._ss_bootstrap_top_box_ang,
        ) <= 0.0:
            raise ValueError(
                "STACK_BOOTSTRAP_TOP stability thresholds must be positive"
            )
        if not 0.0 < self._ss_bootstrap_top_upright_deg <= 90.0:
            raise ValueError("STACK_BOOTSTRAP_TOP_UPRIGHT_DEG must be in (0, 90]")
        if self._ss_humanoid_stability_w < 0.0:
            raise ValueError("STACK_HUMANOID_STABILITY_W must be non-negative")
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
        if self._ss_top_parallel_deg <= 0.0 or self._ss_top_parallel_deg > 180.0:
            raise ValueError("STACK_TOP_PARALLEL_DEG must be in (0, 180]")
        if self._ss_shared_wait_dist <= 0.0:
            raise ValueError("STACK_SHARED_WAIT_DIST must be positive")
        if self._ss_shared_path_clearance < 0.0:
            raise ValueError("STACK_SHARED_PATH_CLEARANCE must be non-negative")
        if self._ss_shared_path_candidates < 2:
            raise ValueError("STACK_SHARED_PATH_CANDIDATES must be at least 2")
        if self._ss_shared_path_retries < 1:
            raise ValueError("STACK_SHARED_PATH_RETRIES must be at least 1")
        if self._ss_shared_path_start_margin < 0.0:
            raise ValueError("STACK_SHARED_PATH_START_MARGIN must be non-negative")
        if min(self._ss_top_wait_reward_w, self._ss_base_hold_reward_w) < 0.0:
            raise ValueError("STACK TOP_WAIT/BASE_HOLD reward weights must be non-negative")
        if self._ss_shared_goal_carry and self._ss_top_wait_at_start:
            raise ValueError(
                "STACK_SHARED_GOAL_CARRY cannot use STACK_TOP_WAIT_AT_START"
            )
        if self._ss_shared_goal_carry and not (
            self._ss_require_staged and self._ss_top_commit_goal
        ):
            raise ValueError(
                "STACK_SHARED_GOAL_CARRY requires STACK_REQUIRE_STAGED=1 "
                "and STACK_TOP_COMMIT_GOAL=1"
            )
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
               self._ss_clear_stall_pen_w, self._ss_clear_reverse_pen_w,
               self._ss_clear_track_pen_w, self._ss_clear_facing_w) < 0.0:
            raise ValueError("negative CLEAR reward weights must be non-negative")
        if self._ss_clear_move_min_frac <= 0.0 or self._ss_clear_move_min_frac > 1.0:
            raise ValueError("STACK_CLEAR_MOVE_MIN_FRAC must be in (0, 1]")
        if self._ss_clear_grace_steps < 0:
            raise ValueError("STACK_CLEAR_GRACE_STEPS must be non-negative")
        if self._ss_stop_hold_steps < 0:
            raise ValueError("STACK_STOP_HOLD_STEPS must be non-negative")
        if min(self._ss_stop_decel_dist, self._ss_stop_lin,
               self._ss_stop_ang, self._ss_stop_contact_force) <= 0.0:
            raise ValueError("STACK stop thresholds must be positive")
        if self._ss_stop_reward_w < 0.0:
            raise ValueError("STACK_STOP_REWARD_W must be non-negative")
        if self._ss_stop_upright_deg <= 0.0 or self._ss_stop_upright_deg > 90.0:
            raise ValueError("STACK_STOP_UPRIGHT_DEG must be in (0, 90]")
        if self._ss_trace_path and min(
            self._ss_trace_events, self._ss_trace_steps
        ) <= 0:
            raise ValueError("STACK_TRACE_EVENTS/STEPS must be positive")
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

        if self._ss_bootstrap_import is not None:
            self._ss_bootstrap_frac = 1.0
            self._ss_bootstrap_eval = True
            self._ss_rehearsal_frac = 0.0
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        if self._ss_bootstrap_import is not None:
            stack_bootstrap.import_bank(self, self._ss_bootstrap_import)
        self._ss_trace_actions = None
        self._ss_trace_envs = {}
        self._ss_trace_next_event = 0
        self._ss_trace_records = []
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
        if self._ss_bootstrap_import is not None:
            bank = self._ss_bootstrap_import
            box_sizes[:] = bank["box_sizes"].reshape(-1, 3).to(box_sizes.device)
            self._box_lib._build_box_bps()
            self._ss_base_agent_np = bank["base_agent"].numpy()
            return super()._load_box_asset(box_sizes)
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
        self._ss_shared_path_min = torch.full(
            (num_envs,), float("nan"), device=self.device
        )
        self._ss_latched_base = torch.zeros((num_envs, 3), device=self.device)
        self._ss_top_goal_committed = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_retreat_goal = torch.zeros((num_envs, 3), device=self.device)
        self._ss_retreat_dir = torch.zeros((num_envs, 2), device=self.device)
        self._ss_clear_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_retreat_arrived = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_stop_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_release_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_prev_release_h = torch.zeros(num_envs, device=self.device)
        self._ss_top_release_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_top_wait_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_stack_age = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_prev_top_h = torch.zeros(num_envs, device=self.device)
        self._ss_prev_top_dist = torch.zeros(num_envs, device=self.device)
        self._ss_bootstrap_active = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_bootstrap_top_balance_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._ss_bootstrap_tick = 0
        self._ss_total_reset_count = torch.zeros((), device=self.device)
        self._ss_bootstrap_eligible_reset_count = torch.zeros(
            (), device=self.device
        )
        self._ss_bootstrap_reset_draw_count = torch.zeros((), device=self.device)
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
            f"retreat_random={int(self._ss_retreat_random)} "
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
            f"handz{int(self._ss_stage_use_hand_z)}/"
            f"zero{int(self._ss_stage_force_zero)}/"
            f"{self._ss_stage_hold_steps}/req{int(self._ss_require_staged)} "
            f"top_wait={int(self._ss_top_wait_at_start)}/"
            f"commit{int(self._ss_top_commit_goal)} "
            f"shared_goal={int(self._ss_shared_goal_carry)}/"
            f"gate{self._ss_shared_wait_dist:.2f}m/"
            f"path_clear{self._ss_shared_path_clearance:.2f}/"
            f"k{self._ss_shared_path_candidates}/"
            f"retry{self._ss_shared_path_retries}/"
            f"start_margin{self._ss_shared_path_start_margin:.2f}/"
            f"wait_r{self._ss_top_wait_reward_w:.2f}/"
            f"base_hold_r{self._ss_base_hold_reward_w:.2f} "
            f"budget={self._ss_pre_steps}+{self._ss_phase_steps} "
            f"bootstrap={self._ss_bootstrap_frac:.2f}/"
            f"wait{int(self._ss_bootstrap_keep_wait)}/"
            f"capture_stable{int(self._ss_bootstrap_capture_stable)}/"
            f"top_balance{self._ss_bootstrap_top_balance_steps}/"
            f"{self._ss_bootstrap_top_root_lin:.2f}/"
            f"{self._ss_bootstrap_top_root_ang:.2f}/"
            f"{self._ss_bootstrap_top_box_lin:.2f}/"
            f"{self._ss_bootstrap_top_box_ang:.2f}/"
            f"grasp{int(self._ss_bootstrap_top_require_grasp)}/"
            f"eval{int(self._ss_bootstrap_eval)} "
            f"bootstrap_loop={self._ss_bootstrap_loop_radius:.2f}/"
            f"{self._ss_bootstrap_loop_scale:.2f}/"
            f"{self._ss_bootstrap_loop_track_w:.2f} "
            f"humanoid_stability_w={self._ss_humanoid_stability_w:.2f} "
            f"top_scale={self._ss_top_scale:.2f} "
            f"above_bonus={self._ss_above_bonus:.2f} "
            f"top_release={self._ss_top_release_progress_w:.2f}/"
            f"{self._ss_top_hold_pen_w:.2f}/"
            f"{self._ss_top_hold_grace_steps} "
            f"top_approach={self._ss_top_approach_progress_w:.2f} "
            f"top_premature={self._ss_top_premature_release_pen_w:.2f} "
            f"top_settle={self._ss_top_settle_steps} "
            f"top_parallel={self._ss_top_parallel_deg:.1f}deg "
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
            f"top_carry_target_only={int(self._ss_debug_top_carry_target_only)} "
            f"top_direct_carry_reward={int(self._ss_top_direct_carry_reward)} "
            f"negative_clear_reward={int(self._ss_negative_clear_reward)} "
            f"negative_clear_w={self._ss_clear_move_w:.2f}/"
            f"{self._ss_clear_hand_pen_w:.2f}/"
            f"{self._ss_clear_stall_pen_w:.2f}/"
            f"{self._ss_clear_reverse_pen_w:.2f} "
            f"track_pen_w={self._ss_clear_track_pen_w:.2f} "
            f"facing_w={self._ss_clear_facing_w:.2f} "
            f"move_min_frac={self._ss_clear_move_min_frac:.2f} "
            f"stop={self._ss_stop_hold_steps}/{self._ss_stop_decel_dist:.2f}m/"
            f"{self._ss_stop_lin:.2f}/{self._ss_stop_ang:.2f}/"
            f"{self._ss_stop_upright_deg:.1f}deg/{self._ss_stop_reward_w:.2f} "
            f"clear_grace={self._ss_clear_grace_steps}",
            flush=True,
        )
        if self._ss_trace_path:
            print(
                f"[stack-trace] path={self._ss_trace_path} "
                f"events={self._ss_trace_events} steps={self._ss_trace_steps} "
                f"keep_wait={int(self._ss_debug_keep_wait)} "
                f"carry_live_on_stack={int(self._ss_debug_carry_live_on_stack)} "
                f"top_carry_target_only={int(self._ss_debug_top_carry_target_only)} "
                f"require_top_balanced={int(self._ss_debug_require_top_balanced)}",
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

    def _top_goal_from_base(self, env_ids):
        base_rows, top_rows = self._role_rows(env_ids)
        boxes = self.humanoid_rows(self._box_states)
        base_size = self._box_lib._box_size[base_rows]
        top_size = self._box_lib._box_size[top_rows]
        goal = boxes[base_rows, 0:3].clone()
        goal[:, 2] = (
            boxes[base_rows, 2]
            + 0.5 * (base_size[:, 2] + top_size[:, 2])
        )
        return goal

    def _commit_top_goal(self, env_ids):
        if len(env_ids) == 0:
            return
        pending = ~self._ss_top_goal_committed[env_ids]
        if not bool(pending.any()):
            return
        ids = env_ids[pending]
        self._ss_top_goal[ids] = self._top_goal_from_base(ids)
        self._ss_top_goal_committed[ids] = True

    def _init_stack_bootstrap_buffers(self):
        if self._ss_bootstrap_frac <= 0.0 or hasattr(self, "_ss_bootstrap_valid"):
            return
        self._ss_bootstrap_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ss_bootstrap_roots = torch.zeros_like(self._humanoid_root_states)
        self._ss_bootstrap_dof_pos = torch.zeros_like(self._dof_pos)
        self._ss_bootstrap_dof_vel = torch.zeros_like(self._dof_vel)
        self._ss_bootstrap_body_states = torch.zeros_like(
            self._initial_humanoid_rigid_body_states
        )
        self._ss_bootstrap_boxes = torch.zeros(
            (self.num_envs, self.num_agents, 13), device=self.device
        )
        self._ss_bootstrap_base_goal = torch.zeros_like(self._ss_base_goal)
        self._ss_bootstrap_stage_goal = torch.zeros_like(self._ss_stage_goal)
        self._ss_bootstrap_top_goal = torch.zeros_like(self._ss_top_goal)
        self._ss_bootstrap_top_goal_committed = torch.zeros_like(
            self._ss_top_goal_committed
        )
        self._ss_bootstrap_latched_base = torch.zeros_like(self._ss_latched_base)
        self._ss_bootstrap_retreat_goal = torch.zeros_like(self._ss_retreat_goal)
        self._ss_bootstrap_retreat_dir = torch.zeros_like(self._ss_retreat_dir)

    def _capture_stack_bootstrap(self, env_ids):
        if self._ss_bootstrap_import is not None:
            return
        if self._ss_bootstrap_frac <= 0.0 or len(env_ids) == 0:
            return
        if self._ss_bootstrap_top_balance_steps > 0:
            env_ids = env_ids[
                self._ss_bootstrap_top_balance_count[env_ids]
                >= self._ss_bootstrap_top_balance_steps
            ]
            if len(env_ids) == 0:
                return
        self._init_stack_bootstrap_buffers()
        self._ss_bootstrap_roots[env_ids] = self._humanoid_root_states[env_ids]
        self._ss_bootstrap_dof_pos[env_ids] = self._dof_pos[env_ids]
        self._ss_bootstrap_dof_vel[env_ids] = self._dof_vel[env_ids]
        self._ss_bootstrap_body_states[env_ids] = torch.cat((
            self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids],
            self._rigid_body_vel[env_ids], self._rigid_body_ang_vel[env_ids],
        ), dim=-1)
        self._ss_bootstrap_boxes[env_ids] = self.agent_axis(self._box_states)[env_ids]
        self._ss_bootstrap_base_goal[env_ids] = self._ss_base_goal[env_ids]
        self._ss_bootstrap_stage_goal[env_ids] = self._ss_stage_goal[env_ids]
        if self._ss_top_commit_goal:
            self._ss_bootstrap_top_goal[env_ids] = self._top_goal_from_base(env_ids)
            self._ss_bootstrap_top_goal_committed[env_ids] = True
        else:
            self._ss_bootstrap_top_goal[env_ids] = self._ss_top_goal[env_ids]
            self._ss_bootstrap_top_goal_committed[env_ids] = (
                self._ss_top_goal_committed[env_ids]
            )
        self._ss_bootstrap_latched_base[env_ids] = self._ss_latched_base[env_ids]
        self._ss_bootstrap_retreat_goal[env_ids] = self._ss_retreat_goal[env_ids]
        self._ss_bootstrap_retreat_dir[env_ids] = self._ss_retreat_dir[env_ids]
        self._ss_bootstrap_valid[env_ids] = True
        if self._ss_bootstrap_save:
            stack_bootstrap.save_bank(
                self._ss_bootstrap_save, self.get_stack_bootstrap_state()
            )
            self._ss_bootstrap_export_done = True

    def get_stack_bootstrap_state(self):
        return stack_bootstrap.export_bank(self)

    def _use_default_amp_history(self, env_ids):
        for task_name, skill_envs in self._reset_ref_env_ids.items():
            for skill_name, ref_ids in skill_envs.items():
                keep = ~torch.isin(ref_ids, env_ids)
                skill_envs[skill_name] = ref_ids[keep]
                row_keep = keep.repeat_interleave(self.num_agents)
                self._reset_ref_motion_ids[task_name][skill_name] = (
                    self._reset_ref_motion_ids[task_name][skill_name][row_keep]
                )
                self._reset_ref_motion_times[task_name][skill_name] = (
                    self._reset_ref_motion_times[task_name][skill_name][row_keep]
                )
        if len(self._reset_default_env_ids) == 0:
            self._reset_default_env_ids = env_ids
        else:
            current = torch.as_tensor(
                self._reset_default_env_ids, dtype=torch.long, device=self.device
            )
            self._reset_default_env_ids = torch.unique(torch.cat((current, env_ids)))

    def _restore_stack_bootstrap(self, env_ids):
        if self._ss_bootstrap_frac <= 0.0 or len(env_ids) == 0:
            return
        if self._is_eval and not self._ss_bootstrap_eval:
            return
        self._init_stack_bootstrap_buffers()
        eligible = env_ids[
            self._ss_bootstrap_valid[env_ids] & (~self._ss_rehearsal[env_ids])
        ]
        if len(eligible) == 0:
            return
        self._ss_bootstrap_eligible_reset_count += len(eligible)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._ss_seed + 100000 + self._ss_bootstrap_tick)
        self._ss_bootstrap_tick += 1
        use = torch.rand(len(eligible), generator=generator) < self._ss_bootstrap_frac
        ids = eligible[use.to(self.device)]
        if len(ids) == 0:
            return
        self._ss_bootstrap_reset_draw_count += len(ids)

        rows = self.agent_rows(ids)
        base_rows, top_rows = self._role_rows(ids)
        self._humanoid_root_states[ids] = self._ss_bootstrap_roots[ids]
        self._dof_pos[ids] = self._ss_bootstrap_dof_pos[ids]
        self._dof_vel[ids] = self._ss_bootstrap_dof_vel[ids]
        self._kinematic_humanoid_rigid_body_states[rows] = (
            self._ss_bootstrap_body_states[ids].reshape(-1, self.num_bodies, 13)
        )
        self.agent_axis(self._box_states)[ids] = self._ss_bootstrap_boxes[ids]
        self._ss_base_goal[ids] = self._ss_bootstrap_base_goal[ids]
        self._ss_stage_goal[ids] = self._ss_bootstrap_stage_goal[ids]
        self._ss_top_goal[ids] = self._ss_bootstrap_top_goal[ids]
        self._ss_top_goal_committed[ids] = (
            self._ss_bootstrap_top_goal_committed[ids]
        )
        self._ss_latched_base[ids] = self._ss_bootstrap_latched_base[ids]
        self._ss_retreat_goal[ids] = self._ss_bootstrap_retreat_goal[ids]
        self._ss_retreat_dir[ids] = self._ss_bootstrap_retreat_dir[ids]

        self._ss_phase[ids] = self.STACK
        self._ss_staged[ids] = True
        self._ss_stack_age[ids] = 0
        self._ss_top_place_count[ids] = 0
        self._ss_top_count[ids] = 0
        self._ss_top_settled[ids] = False
        self._ss_bootstrap_active[ids] = True
        self._ss_dbg_pick_step[rows] = 0
        self._ss_dbg_place_step[base_rows] = 0
        self._ss_dbg_release_step[base_rows] = 0
        self._ss_dbg_clear_step[base_rows] = 0
        self._ss_dbg_staged_step[ids] = 0

        boxes = self.humanoid_rows(self._box_states)
        self._box_tar_pos[base_rows] = self._ss_base_goal[ids]
        if self._ss_bootstrap_keep_wait:
            if self._ss_bootstrap_loop_radius > 0.0:
                self._box_tar_pos[top_rows] = self._ss_top_goal[ids]
                self._set_bootstrap_loop(base_rows)
                self._set_bootstrap_loop(top_rows)
            else:
                self._box_tar_pos[top_rows] = boxes[top_rows, 0:3]
                self._hold_steer(base_rows)
                self._hold_steer(top_rows)
        else:
            if self._ss_top_commit_goal:
                target = self._ss_top_goal[ids]
            else:
                base_size = self._box_lib._box_size[base_rows]
                top_size = self._box_lib._box_size[top_rows]
                target = boxes[base_rows, 0:3].clone()
                target[:, 2] = boxes[base_rows, 2] + 0.5 * (
                    base_size[:, 2] + top_size[:, 2]
                )
            self._box_tar_pos[top_rows] = target
            self._hold_steer(base_rows)
            self._reset_steer(top_rows)
            self._mscale[top_rows] = self._ss_top_scale

        start_dist = (
            boxes[top_rows, 0:3] - self._box_tar_pos[top_rows]
        ).norm(dim=-1)
        self._ss_prev_top_dist[ids] = start_dist
        self._ss_prev_top_h[ids] = 0.0
        self._ss_dbg_stack_dist_start[ids] = start_dist
        self._ss_dbg_stack_dist_min[ids] = start_dist
        self._ss_dbg_z0[rows] = boxes[rows, 2]
        self._every_env_init_dof_pos[rows] = self.humanoid_rows(self._dof_pos)[rows]

        humanoid_actor_ids = self._humanoid_actor_ids_per_env[ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(humanoid_actor_ids),
            len(humanoid_actor_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_actor_ids),
            len(humanoid_actor_ids),
        )
        self._sync_stack_actors(ids)
        self._refresh_sim_tensors()
        self._reset_progress_ref(ids)
        self._use_default_amp_history(ids)

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
        self._finish_stack_trace(env_ids)
        super()._post_object_reset(env_ids)
        if len(env_ids) == 0:
            return
        self._ss_total_reset_count += len(env_ids)
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
        self._ss_top_wait_steps[env_ids] = 0
        self._ss_stack_age[env_ids] = 0
        self._ss_prev_top_h[env_ids] = 0.0
        self._ss_prev_top_dist[env_ids] = 0.0
        self._ss_bootstrap_active[env_ids] = False
        self._ss_bootstrap_top_balance_count[env_ids] = 0
        self._ss_top_goal_committed[env_ids] = False
        self._ss_success[env_ids] = False
        self._ss_failed[env_ids] = False
        self._ss_latched_base[env_ids] = 0.0
        self._ss_retreat_goal[env_ids] = 0.0
        self._ss_retreat_dir[env_ids] = 0.0
        self._ss_clear_age[env_ids] = 0
        self._ss_retreat_arrived[env_ids] = False
        self._ss_stop_count[env_ids] = 0
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

        if self._ss_shared_goal_carry:
            # The task-level XY goal is shared.  Before the base is clear, the
            # top carrier uses a safety gate on its direct route to that goal,
            # so it carries immediately but cannot occupy the unsupported stack
            # pose.  The final target is committed once from the settled base.
            top_start = boxes[env_ids, top_a, 0:2]
            toward = goal_xy - top_start
            distance = toward.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(toward)
            fallback[:, 0] = 1.0
            direction = torch.where(
                distance > 1e-4,
                toward / distance.clamp(min=1e-4),
                fallback,
            )
            gate_offset = torch.minimum(
                torch.full_like(distance[:, 0], self._ss_shared_wait_dist),
                (distance[:, 0] - 0.50).clamp(min=0.0),
            )
            self._ss_stage_goal[env_ids, 0:2] = (
                goal_xy - direction * gate_offset[:, None]
            )
            self._ss_stage_goal[env_ids, 2] = torch.maximum(
                self._ss_top_goal[env_ids, 2],
                torch.full_like(gate_offset, self._ss_stage_z),
            )
        elif self._ss_top_wait_at_start:
            self._ss_stage_goal[env_ids] = boxes[env_ids, top_a, 0:3]
        else:
            side = torch.where(
                top_a == 0, -torch.ones_like(top_a), torch.ones_like(top_a)
            ).float()
            self._ss_stage_goal[env_ids, 0] = goal_xy[:, 0]
            self._ss_stage_goal[env_ids, 1] = goal_xy[:, 1] + side * self._ss_stage_dist
            if self._ss_stage_use_hand_z:
                rigid = self.humanoid_rows(self._rigid_body_pos)
                hands = rigid[top_rows][:, self._key_body_ids[[0, 1]]].mean(dim=1)
                self._ss_stage_goal[env_ids, 2] = hands[:, 2]
            else:
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
        self._ss_shared_path_min[env_ids] = float("nan")
        if self._ss_shared_goal_carry and self._ss_shared_path_clearance > 0.0:
            self._reset_shared_goal_paths(env_ids, base_rows, top_rows)
        else:
            self._reset_steer(self.agent_rows(env_ids))
        self._mscale[base_rows] = 1.0
        self._mscale[top_rows] = 1.0
        self._reset_progress_ref(env_ids)
        self._restore_stack_bootstrap(env_ids)
        return

    # ------------------------------------------------------------- debug trace

    def pre_physics_step(self, actions):
        if self._ss_bootstrap_save_once:
            self._ss_bootstrap_collect_tick += 1
        if self._ss_trace_path:
            self._ss_trace_actions = actions.detach().clone()
        return super().pre_physics_step(actions)

    def _claim_stack_trace(self, env_ids):
        if not self._ss_trace_path:
            return env_ids[:0]
        claimed = []
        for env_id in env_ids.detach().cpu().tolist():
            if env_id in self._ss_trace_envs:
                claimed.append(env_id)
                continue
            if self._ss_trace_next_event >= self._ss_trace_events:
                continue
            self._ss_trace_envs[env_id] = self._ss_trace_next_event
            self._ss_trace_next_event += 1
            claimed.append(env_id)
        return torch.as_tensor(claimed, dtype=torch.long, device=self.device)

    def _record_stack_trace(self, env_ids, marker):
        if not self._ss_trace_path or len(env_ids) == 0:
            return
        keep = [
            int(env_id) in self._ss_trace_envs
            for env_id in env_ids.detach().cpu().tolist()
        ]
        env_ids = env_ids[
            torch.as_tensor(keep, dtype=torch.bool, device=self.device)
        ]
        if len(env_ids) == 0:
            return

        rows = self.agent_rows(env_ids)
        env = torch.div(rows, self.num_agents, rounding_mode="floor")
        agent = rows % self.num_agents
        roots = self.humanoid_rows(self._humanoid_root_states)[rows]
        boxes = self.humanoid_rows(self._box_states)[rows]
        task_obs = self._compute_task_obs(env_ids)
        teammate_dim = TEAMMATE_DIM * (self.num_agents - 1)
        steer_end = teammate_dim + self.steer_dim()
        carry_dim = CARRY_HI - CARRY_LO

        contact = self.humanoid_rows(self._contact_forces)[rows].clone()
        contact[:, self._contact_body_ids] = 0
        fall_contact = (contact.abs() > 0.1).any(dim=-1).any(dim=-1)
        body_height = self.humanoid_rows(self._rigid_body_pos)[rows, :, 2]
        fall_height = body_height < self._termination_heights
        fall_height[:, self._contact_body_ids] = False
        fall_height = fall_height.any(dim=-1)

        if self._ss_trace_actions is None:
            action_l2 = torch.full_like(roots[:, 2], float("nan"))
            action_max = action_l2.clone()
        else:
            actions = self._ss_trace_actions[rows]
            action_l2 = actions.norm(dim=-1)
            action_max = actions.abs().amax(dim=-1)

        def norm(lo, hi):
            return task_obs[:, lo:hi].norm(dim=-1)

        event = torch.as_tensor(
            [
                self._ss_trace_envs[int(value)]
                for value in env.detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=self.device,
        )
        is_base = agent == self._ss_base_agent[env]
        target = self._box_tar_pos[rows]
        values = torch.stack((
            event,
            env.float(),
            agent.float(),
            is_base.float(),
            torch.full_like(event, float(marker)),
            self.progress_rows()[rows].float(),
            self._ss_phase[env].float(),
            self._ss_stack_age[env].float(),
            self.reset_buf[env].float(),
            self._terminate_buf[env].float(),
            action_l2,
            action_max,
            self.obs_buf[rows].norm(dim=-1),
            norm(0, teammate_dim),
            norm(teammate_dim, steer_end),
            norm(steer_end, steer_end + carry_dim),
            norm(steer_end + carry_dim, steer_end + 2 * carry_dim),
            self._m_at(self._arc_root[rows], rows) / 1.6,
            roots[:, 2],
            roots[:, 7:10].norm(dim=-1),
            roots[:, 10:13].norm(dim=-1),
            self._upright_score(roots[:, 3:7]),
            fall_contact.float(),
            fall_height.float(),
            (fall_contact & fall_height).float(),
            (boxes[:, 0:3] - target).norm(dim=-1),
            boxes[:, 7:10].norm(dim=-1),
            self._hand_surface_distances(rows).amin(dim=-1),
            self._hand_surface_distances(rows).amax(dim=-1),
        ), dim=-1)
        self._ss_trace_records.append(values.detach().cpu().numpy())
        if len(self._ss_trace_records) % 10 == 0:
            self._save_stack_trace()

    def _save_stack_trace(self):
        if not self._ss_trace_records:
            return
        names = np.asarray((
            "event", "env", "agent", "is_base", "marker", "progress",
            "phase", "stack_age", "reset", "terminate_env", "action_l2",
            "action_max", "obs_l2", "teammate_l2", "steer_l2",
            "carry0_l2", "carry1_l2", "cmd_speed", "root_z", "root_speed",
            "root_ang_speed", "root_upright", "fall_contact", "fall_height",
            "fallen", "box_target_dist", "box_speed", "hand_min", "hand_max",
        ))
        path = os.path.abspath(self._ss_trace_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle, names=names, data=np.concatenate(self._ss_trace_records)
            )
        os.replace(tmp, path)

    def _finish_stack_trace(self, env_ids):
        if not self._ss_trace_path or not hasattr(self, "_ss_trace_envs"):
            return
        changed = False
        for env_id in env_ids.detach().cpu().tolist():
            if int(env_id) in self._ss_trace_envs:
                del self._ss_trace_envs[int(env_id)]
                changed = True
        if changed:
            self._save_stack_trace()

    def _record_active_stack_trace(self):
        if not self._ss_trace_path or not self._ss_trace_envs:
            return
        env_ids = torch.as_tensor(
            list(self._ss_trace_envs), dtype=torch.long, device=self.device
        )
        age = self._ss_stack_age[env_ids]
        keep = (age > 0) & (age <= self._ss_trace_steps)
        self._record_stack_trace(env_ids[keep], 1)

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

    @staticmethod
    def _box_parallel_error_deg(base_quat, top_quat):
        """Return face-alignment error, treating quarter turns as parallel.

        A cube rotated 90 degrees still has parallel faces, while the unstable
        45-degree diamond placement does not. The normal-axis term also rejects
        a top box whose upper/lower face is tilted relative to the base box.
        """
        axes = torch.eye(3, device=base_quat.device, dtype=base_quat.dtype)
        axes = axes.unsqueeze(0).expand(len(base_quat), -1, -1)
        base_axes = quat_rotate(
            base_quat[:, None, :].expand(-1, 3, -1).reshape(-1, 4),
            axes.reshape(-1, 3),
        ).view(-1, 3, 3)
        top_axes = quat_rotate(
            top_quat[:, None, :].expand(-1, 3, -1).reshape(-1, 4),
            axes.reshape(-1, 3),
        ).view(-1, 3, 3)

        # Each horizontal top-box edge may align with either horizontal base
        # edge. Absolute dot products make 180-degree flips equivalent.
        horizontal = torch.abs(
            torch.einsum("nai,nbi->nab", top_axes[:, :2], base_axes[:, :2])
        )
        top_x = horizontal[:, 0].amax(dim=-1)
        top_y = horizontal[:, 1].amax(dim=-1)
        normal = torch.abs((top_axes[:, 2] * base_axes[:, 2]).sum(dim=-1))
        alignment = torch.minimum(normal, torch.minimum(top_x, top_y))
        return torch.rad2deg(torch.acos(alignment.clamp(-1.0, 1.0)))

    def _base_stop_features(self, base_rows):
        roots = self.humanoid_rows(self._humanoid_root_states)[base_rows]
        root_speed = roots[:, 7:9].norm(dim=-1)
        root_ang = roots[:, 10:13].norm(dim=-1)
        upright = self._upright_score(roots[:, 3:7])
        contacts = self.humanoid_rows(self._contact_forces)[base_rows]
        foot_force = contacts[:, self._key_body_ids[[2, 3]]].norm(dim=-1)
        double_support = (
            foot_force >= self._ss_stop_contact_force
        ).all(dim=-1)
        stable = (
            (root_speed <= self._ss_stop_lin)
            & (root_ang <= self._ss_stop_ang)
            & (
                upright
                >= math.cos(math.radians(self._ss_stop_upright_deg))
            )
            & double_support
        )
        return root_speed, root_ang, upright, double_support, stable

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

    def _direct_carry_reward(self, rows):
        """Juan carry reward evaluated against the live carry target.

        This deliberately bypasses only the path/steering reward.  The target,
        carry observations, power penalty, and all later sequential shaping stay
        unchanged, so it can be selected for A2 in STACK without touching A1.
        """
        humanoid = self.humanoid_rows(self._humanoid_root_states)[rows]
        rigid_body_pos = self.humanoid_rows(self._rigid_body_pos)[rows]
        box = self.humanoid_rows(self._box_states)[rows]
        root_pos = humanoid[:, 0:3]
        box_pos = box[:, 0:3]
        target = self._box_tar_pos[rows]
        hands_ids = self._key_body_ids[[0, 1]]

        reward = compute_walk_reward(
            root_pos,
            self._prev_root_pos[rows],
            box_pos,
            self.dt,
            1.5,
            self._carry_rwd_only_vel_reward,
        )
        reward += compute_carry_reward(
            box_pos,
            self._prev_box_pos[rows],
            target,
            self.dt,
            1.5,
            self._box_lib._box_size[rows],
            self._carry_rwd_only_vel_reward,
            self._carry_rwd_box_vel_penalty,
            self._carry_rwd_box_vel_pen_coeff,
            self._carry_rwd_box_vel_pen_thre,
        )
        reward += compute_handheld_reward(
            rigid_body_pos,
            box_pos,
            hands_ids,
            target,
            self._carry_rwd_only_height_handheld_reward,
        )
        reward += compute_putdown_reward(box_pos, target)
        if self._power_reward:
            power = torch.abs(
                self.dof_force_tensor[rows]
                * self.humanoid_rows(self._dof_vel)[rows]
            ).sum(dim=-1)
            reward -= self._power_coefficient * power
        return reward

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
            "top_hold_penalty", "top_wait", "top_total", "base_stability",
            "top_stability", "base_hold", "base_loop_tracking", "stop_decel",
        ):
            self.extras[f"tb/stack_reward/{key}"] = nan
        for key in (
            "top_distance", "top_above", "top_grasp_ok", "top_settled",
            "top_release_ready", "top_hand_factor", "top_grasp_factor", "top_hand_clear",
            "top_stack_age", "top_waiting", "top_wait_steps",
            "base_hold_anchor_error", "base_hold_speed_score",
            "stop_decel_active", "stop_target_speed", "stop_root_speed",
            "stop_root_ang", "stop_double_support", "stop_stable",
            "stop_streak_fraction",
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
            or self._ss_top_wait_reward_w > 0.0
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
        stack_phase = self._ss_phase == self.STACK
        if self._ss_top_direct_carry_reward and bool(stack_phase.any()):
            direct_carry = self._direct_carry_reward(top_rows)
            self.rew_buf[top_rows] = torch.where(
                stack_phase, direct_carry, self.rew_buf[top_rows]
            )
            top_native = torch.where(stack_phase, direct_carry, top_native)
        top_target = self._box_tar_pos[top_rows]
        top_xy_err = (top[:, 0:2] - top_target[:, 0:2]).norm(dim=-1)
        top_z_err = (top[:, 2] - top_target[:, 2]).abs()
        top_dist = torch.sqrt(top_xy_err.square() + top_z_err.square())
        roots = self.humanoid_rows(self._humanoid_root_states)
        base_roots = roots[base_rows]
        top_roots = roots[top_rows]

        def humanoid_stability(root):
            height = torch.clamp((root[:, 2] - 0.45) / 0.40, 0.0, 1.0)
            angular = torch.exp(-0.25 * root[:, 10:13].square().sum(dim=-1))
            return self._upright_score(root[:, 3:7]) * height * angular

        base_stability = humanoid_stability(base_roots)
        top_stability = humanoid_stability(top_roots)
        base_stability_reward = (
            self._ss_humanoid_stability_w
            * base_stability
            * stack_phase.float()
        )
        top_stability_reward = (
            self._ss_humanoid_stability_w
            * top_stability
            * stack_phase.float()
        )
        self.rew_buf[top_rows] += top_stability_reward
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
        if self._ss_shared_goal_carry:
            top_waiting = (
                (~self._ss_rehearsal)
                & (self._ss_phase < self.STACK)
                & self._ss_staged
                & top_grasp_ok
                & (top_dist <= 1.5 * self._ss_stage_tol)
            )
        else:
            top_waiting = torch.zeros_like(stack_phase)
        top_box_stability = torch.exp(
            -10.0 * top[:, 7:10].square().sum(dim=-1)
            -0.5 * top[:, 10:13].square().sum(dim=-1)
        )
        top_wait_position = torch.exp(-8.0 * top_dist.square())
        top_wait_reward = (
            self._ss_top_wait_reward_w
            * top_waiting.float()
            * (
                0.35 * top_grasp_ok.float()
                + 0.25 * top_wait_position
                + 0.20 * top_box_stability
                + 0.20 * top_stability
            )
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
            + top_wait_reward
            + above_bonus
        )
        self.rew_buf[top_rows] += top_local
        self._ss_prev_top_dist = torch.where(
            stack_phase, top_dist, torch.zeros_like(self._ss_prev_top_dist)
        )
        self._ss_prev_top_h = torch.where(
            release_ready, top_h, torch.zeros_like(self._ss_prev_top_h)
        )
        top_sample = (
            top_hold_required
            | top_waiting
            | stack_phase
            | self._ss_above_bonus_pending
        )
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
            ("top_wait", top_wait_reward),
            ("top_total", self.rew_buf[top_rows]),
        ):
            self.extras[f"tb/stack_reward/{key}"] = torch.where(
                top_sample, value, nan
            )
        self.extras["tb/stack_reward/base_stability"] = torch.where(
            stack_phase, base_stability_reward, nan
        )
        self.extras["tb/stack_reward/top_stability"] = torch.where(
            stack_phase, top_stability_reward, nan
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
            ("top_waiting", top_waiting.float()),
            ("top_wait_steps", self._ss_top_wait_steps.float()),
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
        clear_phase = self._ss_phase == self.CLEAR
        remaining = (
            self._s_end[base_rows] - arc
        ).clamp(min=0.0)
        if self._ss_stop_hold_steps > 0:
            decel_phase = (
                clear_phase
                & (remaining <= self._ss_stop_decel_dist)
            )
        else:
            decel_phase = torch.zeros_like(clear_phase)
        (stop_root_speed, stop_root_ang, stop_upright,
         stop_double_support, stop_stable) = self._base_stop_features(base_rows)
        roots = self.humanoid_rows(self._humanoid_root_states)[base_rows]
        v_along = (roots[:, 7:9] * self._ss_retreat_dir).sum(dim=-1)
        v_lateral = (
            roots[:, 7] * -self._ss_retreat_dir[:, 1]
            + roots[:, 8] * self._ss_retreat_dir[:, 0]
        )
        decel_fraction = torch.clamp(
            remaining / self._ss_stop_decel_dist, 0.0, 1.0
        )
        stop_target_speed = cmd_speed * decel_fraction
        stop_speed_score = torch.exp(
            -((v_along - stop_target_speed) / self._ss_stop_lin).square()
            -(v_lateral / self._ss_stop_lin).square()
        )
        stop_balance_score = (
            stop_upright
            * torch.exp(-(stop_root_ang / self._ss_stop_ang).square())
        )
        stop_reward = (
            self._ss_stop_reward_w
            * decel_phase.float()
            * (
                0.50 * stop_speed_score
                + 0.25 * stop_balance_score
                + 0.25 * stop_double_support.float()
            )
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
        base_hold_anchor_error = (
            roots[:, 0:2] - self._ss_retreat_goal[:, 0:2]
        ).norm(dim=-1)
        base_hold_speed_score = torch.exp(
            -(stop_root_speed / self._ss_stop_lin).square()
            -(stop_root_ang / self._ss_stop_ang).square()
        )
        base_hold_score = (
            0.35 * torch.exp(-8.0 * base_hold_anchor_error.square())
            + 0.25 * base_hold_speed_score
            + 0.20 * stop_upright.clamp(0.0, 1.0)
            + 0.20 * stop_double_support.float()
        )
        base_hold_reward = (
            self._ss_base_hold_reward_w
            * stack_phase.float()
            * base_hold_score
        )
        stack_r = (
            clear_now.float() * (0.40 * support + 0.25 * stable)
            + 0.35 * clear_score
            + base_stability_reward
            + base_hold_reward
        )
        positive_steer = clear_now.float() * path_quality * clear_motion
        clear_positive_steer = positive_steer * (~decel_phase).float()
        classic_speed_match = torch.exp(
            -self.steer_vel_k * (cmd_speed - arc_speed).square()
        )
        classic_speed_match = torch.where(
            arc_speed > 0.0,
            classic_speed_match,
            torch.zeros_like(classic_speed_match),
        )
        classic_path_penalty = self.steer_pos_c * (
            torch.exp(-0.5 * self._lat_root[base_rows].square()) - 1.0
        )
        classic_steer = (
            0.2 * self.steer_vel_w * classic_speed_match + classic_path_penalty
        ) * (~decel_phase).float()
        loop_active = (
            stack_phase
            & self._ss_bootstrap_active
            & (self._ss_bootstrap_loop_radius > 0.0)
        )
        base_loop_tracking = (
            self._ss_bootstrap_loop_track_w
            * positive_steer
            * loop_active.float()
        )
        stack_r = stack_r + base_loop_tracking
        clear_r = (
            ms20_clear_r
            + self._ss_clear_steer_w * classic_steer
            + stop_reward
            - clear_base_penalty
        )
        if self._ss_negative_clear_reward:
            # arc is a forward-only path ratchet, so it cannot distinguish
            # standing still from moving in the wrong direction. Use the actual
            # root velocity projected onto the commanded retreat direction.
            roots = self.humanoid_rows(self._humanoid_root_states)[base_rows]
            direction = self._ss_retreat_dir
            if self._ss_clear_track_pen_w > 0.0 or self._ss_clear_facing_w > 0.0:
                index = (arc / sp.DS).long().clamp(0, sp.V - 2)
                tangent = (
                    self._gt_path[base_rows, index + 1]
                    - self._gt_path[base_rows, index]
                )
                direction = torch.nn.functional.normalize(tangent, dim=-1)
            v_along = (roots[:, 7:9] * direction).sum(dim=-1)
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
            move_active = (~decel_phase).float()
            # CLEAR is entered only after both hands were clear.  Any later
            # recontact removes all positive movement credit and incurs the full
            # recontact penalty, instead of leaving a near-threshold loophole.
            move_reward = (
                self._ss_clear_move_w
                * clear_now.float()
                * path_quality
                * move_ratio
                * move_active
            )
            heading = torch_utils.calc_heading_quat(roots[:, 3:7])
            forward = torch.zeros_like(roots[:, 0:3])
            forward[:, 0] = 1.0
            facing_score = torch.clamp(
                (quat_rotate(heading, forward)[:, 0:2] * direction).sum(dim=-1),
                0.0,
                1.0,
            )
            facing_reward = (
                self._ss_clear_facing_w
                * clear_now.float()
                * facing_score
                * move_active
            )
            hand_penalty = (
                self._ss_clear_hand_pen_w * (~clear_now).float()
            )
            stall_penalty = (
                self._ss_clear_stall_pen_w * stall_ratio * after_grace * move_active
            )
            reverse_penalty = (
                self._ss_clear_reverse_pen_w
                * reverse_ratio * after_grace * move_active
            )
            track_penalty = (
                self._ss_clear_track_pen_w
                * torch.maximum(stall_ratio, 1.0 - path_quality)
                * after_grace * move_active
            )
            clear_r = (
                move_reward
                + facing_reward
                + stop_reward
                - hand_penalty
                - stall_penalty
                - reverse_penalty
                - track_penalty
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
            self.extras["tb/stack_reward/facing_positive"] = tb_clear(facing_reward)
            self.extras["tb/stack_reward/hand_penalty"] = tb_clear(-hand_penalty)
            self.extras["tb/stack_reward/stall_penalty"] = tb_clear(-stall_penalty)
            self.extras["tb/stack_reward/track_penalty"] = tb_clear(-track_penalty)
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
            self.extras["tb/stack_state/facing_score"] = tb_clear(facing_score)
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
        def tb_stop(value):
            return torch.where(clear_phase, value, nan)

        self.extras["tb/stack_reward/stop_decel"] = tb_stop(stop_reward)
        self.extras["tb/stack_state/stop_decel_active"] = tb_stop(
            decel_phase.float()
        )
        self.extras["tb/stack_state/stop_target_speed"] = tb_stop(stop_target_speed)
        self.extras["tb/stack_state/stop_root_speed"] = tb_stop(stop_root_speed)
        self.extras["tb/stack_state/stop_root_ang"] = tb_stop(stop_root_ang)
        self.extras["tb/stack_state/stop_double_support"] = tb_stop(
            stop_double_support.float()
        )
        self.extras["tb/stack_state/stop_stable"] = tb_stop(stop_stable.float())
        self.extras["tb/stack_state/stop_streak_fraction"] = tb_stop(
            self._ss_stop_count.float() / max(self._ss_stop_hold_steps, 1)
        )
        self.extras["tb/stack_state/retreat_arrived"] = tb_stop(
            self._ss_retreat_arrived.float()
        )
        self.extras["tb/stack_state/retreat_endpoint_error"] = tb_stop(
            base_hold_anchor_error
        )
        self.extras["tb/stack_reward/classic_steer"] = tb_stop(
            self._ss_clear_steer_w * classic_steer
        )
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
        self.extras["stack_clear_positive_steer"] = clear_positive_steer
        self.extras["stack_clear_classic_steer"] = classic_steer
        self.extras["stack_clear_base_penalty"] = clear_base_penalty

        self.extras["tb/stack_reward/base_hold"] = torch.where(
            stack_phase, base_hold_reward, nan
        )
        self.extras["tb/stack_state/base_hold_anchor_error"] = torch.where(
            stack_phase, base_hold_anchor_error, nan
        )
        self.extras["tb/stack_state/base_hold_speed_score"] = torch.where(
            stack_phase, base_hold_speed_score, nan
        )
        self.extras["tb/stack_reward/base_loop_tracking"] = torch.where(
            stack_phase, base_loop_tracking, nan
        )
        if bool(failed.any()):
            self.rew_buf[base_rows[failed]] = -1.0
        return

    # --------------------------------------------------------------- controller

    def _set_bootstrap_loop(self, rows):
        if len(rows) == 0:
            return
        roots = self.humanoid_rows(self._humanoid_root_states)[rows, 0:2]
        env_ids = rows // self.num_agents
        away_raw = roots - self._ss_base_goal[env_ids, 0:2]
        norm = away_raw.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(away_raw)
        fallback[:, 0] = 1.0
        away = torch.where(
            norm > 1e-4, away_raw / norm.clamp(min=1e-4), fallback
        )
        tangent = torch.stack((-away[:, 1], away[:, 0]), dim=-1)
        radius = self._ss_bootstrap_loop_radius
        center = roots + radius * away
        theta = torch.linspace(0.0, 6.0 * math.pi, 97, device=self.device)
        waypoints = (
            center[:, None, :]
            - radius * torch.cos(theta)[None, :, None] * away[:, None, :]
            + radius * torch.sin(theta)[None, :, None] * tangent[:, None, :]
        )
        path, n_end = sp.resample(waypoints, ds=0.1, total=320, with_end=True)
        boxes = self.humanoid_rows(self._box_states)[rows, 0:2]
        box_arc = sp.project(boxes, path)[0]
        self._gt_path[rows] = path
        self._s_end[rows] = n_end.to(self._s_end.dtype)
        self._arc_root[rows] = 0.0
        self._arc_box[rows] = box_arc
        self._prev_arc[rows] = 0.0
        self._mscale[rows] = self._ss_bootstrap_loop_scale

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

    def _reset_shared_goal_paths(self, env_ids, base_rows, top_rows):
        """Jointly choose a Top staging path that stays clear of Base's path.

        The ms52 path generator and policy interface remain unchanged. Base is
        generated once; Top candidates use the same generator and differ only
        in their waiting point on the shared-goal safety circle. The candidate
        with the largest sampled path-to-path clearance is installed.
        """
        self._reset_steer(base_rows)
        boxes = self.humanoid_rows(self._box_states)
        top_start = boxes[top_rows, 0:2]
        goal = self._ss_base_goal[env_ids, 0:2]
        near = top_start - goal
        distance = near.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(near)
        fallback[:, 0] = 1.0
        near = torch.where(
            distance > 1e-4, near / distance.clamp(min=1e-4), fallback
        )
        side = torch.stack((-near[:, 1], near[:, 0]), dim=-1)
        radius = torch.minimum(
            torch.full_like(distance[:, 0], self._ss_shared_wait_dist),
            (distance[:, 0] - 0.50).clamp(min=0.0),
        )

        best_score = torch.full(
            (len(env_ids),), -1.0, device=self.device
        )
        best_base_path = self._gt_path[base_rows].clone()
        best_base_end = self._s_end[base_rows].clone()
        best_base_box_arc = self._arc_box[base_rows].clone()
        best_base_scale = self._mscale[base_rows].clone()
        best_goal = self._ss_stage_goal[env_ids, 0:2].clone()
        best_path = self._gt_path[top_rows].clone()
        best_end = self._s_end[top_rows].clone()
        best_box_arc = self._arc_box[top_rows].clone()
        best_scale = self._mscale[top_rows].clone()

        for retry in range(self._ss_shared_path_retries):
            active = best_score < self._ss_shared_path_clearance
            if not bool(active.any()):
                break
            active_index = torch.nonzero(active, as_tuple=False).squeeze(-1)
            active_env_ids = env_ids[active]
            active_base_rows = base_rows[active]
            active_top_rows = top_rows[active]
            active_near = near[active]
            active_side = side[active]
            active_goal = goal[active]
            active_radius = radius[active]
            if retry > 0:
                # Some geometries cannot clear the fixed first Base curve no
                # matter which safety-gate point Top uses. Resample only those
                # unresolved Base/Top pairs with the same ms52 generator.
                self._reset_steer(active_base_rows)

            for index in range(self._ss_shared_path_candidates):
                # Offset later retry rings so their gate points and freshly
                # sampled ms52 curves are not exact repeats of the first ring.
                phase = retry / self._ss_shared_path_retries
                angle = (
                    2.0 * math.pi
                    * (index + phase)
                    / self._ss_shared_path_candidates
                )
                direction = (
                    math.cos(angle) * active_near
                    + math.sin(angle) * active_side
                )
                candidate_goal = (
                    active_goal + active_radius[:, None] * direction
                )
                self._ss_stage_goal[active_env_ids, 0:2] = candidate_goal
                self._box_tar_pos[active_top_rows, 0:2] = candidate_goal
                self._reset_steer(active_top_rows)

                clearance = self._sampled_carry_path_clearance(
                    self._gt_path[active_base_rows],
                    self._arc_box[active_base_rows],
                    self._s_end[active_base_rows],
                    self._gt_path[active_top_rows],
                    self._arc_box[active_top_rows],
                    self._s_end[active_top_rows],
                    self._ss_shared_path_start_margin,
                )
                better = clearance > best_score[active_index]
                if bool(better.any()):
                    chosen = active_index[better]
                    chosen_base_rows = active_base_rows[better]
                    chosen_rows = active_top_rows[better]
                    best_score[chosen] = clearance[better]
                    best_base_path[chosen] = self._gt_path[chosen_base_rows]
                    best_base_end[chosen] = self._s_end[chosen_base_rows]
                    best_base_box_arc[chosen] = self._arc_box[chosen_base_rows]
                    best_base_scale[chosen] = self._mscale[chosen_base_rows]
                    best_goal[chosen] = candidate_goal[better]
                    best_path[chosen] = self._gt_path[chosen_rows]
                    best_end[chosen] = self._s_end[chosen_rows]
                    best_box_arc[chosen] = self._arc_box[chosen_rows]
                    best_scale[chosen] = self._mscale[chosen_rows]

        self._gt_path[base_rows] = best_base_path
        self._s_end[base_rows] = best_base_end
        self._arc_root[base_rows] = 0.0
        self._arc_box[base_rows] = best_base_box_arc
        self._prev_arc[base_rows] = 0.0
        self._mscale[base_rows] = best_base_scale
        self._ss_stage_goal[env_ids, 0:2] = best_goal
        self._box_tar_pos[top_rows, 0:2] = best_goal
        self._gt_path[top_rows] = best_path
        self._s_end[top_rows] = best_end
        self._arc_root[top_rows] = 0.0
        self._arc_box[top_rows] = best_box_arc
        self._prev_arc[top_rows] = 0.0
        self._mscale[top_rows] = best_scale
        self._ss_shared_path_min[env_ids] = best_score

    @staticmethod
    def _sampled_carry_path_clearance(
        base_path,
        base_box_arc,
        base_end_arc,
        top_path,
        top_box_arc,
        top_end_arc,
        start_margin=0.0,
        stride=4,
    ):
        """Minimum geometric clearance between the two active carry corridors.

        gen_full_v2 stores person-to-box approach before _arc_box and
        extrapolated padding after _s_end. Neither part is a box carry route,
        so including them makes the score depend on unused path cells.
        start_margin additionally excludes the unavoidable pickup/departure
        neighborhood when the two reset boxes begin closer than the requested
        corridor clearance.
        """
        base_sample = base_path[:, ::stride]
        top_sample = top_path[:, ::stride]
        arc = (
            torch.arange(
                0, base_path.shape[1], stride,
                device=base_path.device,
                dtype=base_path.dtype,
            )
            * sp.DS
        )
        base_valid = (
            (arc[None, :] >= base_box_arc[:, None] + start_margin)
            & (arc[None, :] <= base_end_arc[:, None])
        )
        top_valid = (
            (arc[None, :] >= top_box_arc[:, None] + start_margin)
            & (arc[None, :] <= top_end_arc[:, None])
        )
        pair_valid = base_valid[:, :, None] & top_valid[:, None, :]
        distance = torch.cdist(base_sample, top_sample)
        distance = distance.masked_fill(~pair_valid, float("inf"))
        clearance = distance.amin(dim=(1, 2))
        # A very short route can have no sampled cell after start_margin. Such
        # a candidate is not eligible to claim a safe corridor.
        return torch.where(
            torch.isfinite(clearance), clearance, torch.zeros_like(clearance)
        )

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

        if self._ss_retreat_random:
            g = torch.Generator(device="cpu").manual_seed(
                self.steer_seed + self._steer_tick + int(env_ids[0])
            )
            angle = torch.rand(len(env_ids), 16, generator=g).to(roots) * (math.pi / 2)
            angle = torch.cat((angle, roots.new_full((len(env_ids), 2), math.pi / 2)), dim=1)
            signs = roots.new_ones(18)
            signs[8:16] = -1
            signs[-1] = -1
            candidate_side = side[:, None, :] * signs[None, :, None]
            directions = (
                torch.cos(angle)[..., None] * away[:, None, :]
                + torch.sin(angle)[..., None] * candidate_side
            )
            waypoints = roots[:, None, :] + directions * 0.15
            # Resampling discards up to two DS cells before the clipped endpoint.
            distance = max(self._ss_retreat_dist, 0.8) + 2 * sp.DS
            endpoints = roots[:, None, :] + directions * distance
            top_rows = self._role_rows(env_ids)[1]
            top_box = self.humanoid_rows(self._box_states)[top_rows, :2]
            starts = torch.stack((roots[:, None, :].expand_as(waypoints), waypoints), dim=2)
            ends = torch.stack((waypoints, endpoints), dim=2)
            segment = ends - starts
            offset = top_box[:, None, None, :] - starts
            projection = (
                (offset * segment).sum(dim=-1)
                / segment.square().sum(dim=-1).clamp(min=1e-8)
            ).clamp(0.0, 1.0)
            closest = starts + projection[..., None] * segment
            clearance = (closest - top_box[:, None, None, :]).norm(dim=-1).amin(dim=-1)
            top_radius = (
                0.5 * self._box_lib._box_size[top_rows, :2].norm(dim=-1)
                + self._ss_clear_route_margin
            )
            required = torch.minimum((roots - top_box).norm(dim=-1), top_radius)
            safe = clearance >= required[:, None] - 1e-6
            if not bool(safe.any(dim=-1).all()):
                raise RuntimeError("No box-clear random retreat path for the current state")
            choice = safe.long().argmax(dim=-1)
            batch = torch.arange(len(env_ids), device=roots.device)
            waypoint = waypoints[batch, choice]
            clear = endpoints[batch, choice]
            move_dir = directions[batch, choice]
            lat_max = 0.0
        elif self._ss_clear_route_around:
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
            or self._ss_debug_top_carry_target_only
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
        top_stack = (
            self._ss_debug_top_carry_target_only
            & (~self._ss_rehearsal[env])
            & (agent != self._ss_base_agent[env])
            & (self._ss_phase[env] >= self.STACK)
            & (self._ss_phase[env] <= self.SUCCESS)
        )
        if not (bool(retreat.any()) or bool(top_stack.any())):
            return obs

        out = obs.clone()
        steer_start = TEAMMATE_DIM * (self.num_agents - 1)
        carry_start = steer_start + self.steer_dim()
        carry_dim = CARRY_HI - CARRY_LO
        # Preserve the steering token and attention layout.  This is raw-value
        # zero-padding only, symmetric with the base carry zero-padding below.
        # The top's live carry tokens still contain the newly assigned stack
        # target, while the base row remains byte-for-byte unchanged.
        if bool(top_stack.any()):
            out[top_stack, steer_start:carry_start] = 0.0
        if bool(retreat.any()) and (
            self._ss_zero_carry_obs or self._ss_dynamic_carry_mask
        ):
            # Zero observation and attention masking are deliberately separate.
            # STACK_ZERO_CARRY_OBS leaves both carry token positions active in
            # the frozen Transformer; STACK_DYNAMIC_CARRY_MASK additionally
            # masks those positions in the network.
            zero = retreat
            if self._ss_debug_carry_live_on_stack:
                zero &= self._ss_phase[env] < self.STACK
            out[zero, carry_start:carry_start + 2 * carry_dim] = 0.0
        elif bool(retreat.any()):
            virtual = self._virtual_retreat_carry_obs(rows[retreat], env[retreat])
            out[retreat, carry_start:carry_start + carry_dim] = virtual
            out[retreat, carry_start + carry_dim:carry_start + 2 * carry_dim] = virtual
        return out

    def _update_stack_controller(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        (base_rows, top_rows, base, top, hand_dist, xy_err, z_err,
         _, _, root_dist) = self._base_features()
        top_parallel_error_deg = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )

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
            self._compute_observations()

        # In the legacy staging mode A2 carries its box near the base goal. In
        # Juan-style wait mode the stage is A2's reset box pose, so grasping is
        # intentionally deferred until the committed STACK goal is activated.
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
        stage_ready = top_grasped
        if self._ss_top_wait_at_start:
            stage_ready = torch.ones_like(top_grasped)

        if (
            self._ss_bootstrap_frac > 0.0
            and self._ss_bootstrap_top_balance_steps > 0
        ):
            top_roots = self.humanoid_rows(self._humanoid_root_states)[top_rows]
            top_balance_grasp = top_grasped
            if not self._ss_bootstrap_top_require_grasp:
                top_balance_grasp = torch.ones_like(top_grasped)
            top_bootstrap_balanced = (
                pre_stack
                & top_balance_grasp
                & (
                    top_roots[:, 7:10].norm(dim=-1)
                    <= self._ss_bootstrap_top_root_lin
                )
                & (
                    top_roots[:, 10:13].norm(dim=-1)
                    <= self._ss_bootstrap_top_root_ang
                )
                & (
                    self._upright_score(top_roots[:, 3:7])
                    >= math.cos(math.radians(self._ss_bootstrap_top_upright_deg))
                )
                & (top[:, 7:10].norm(dim=-1) <= self._ss_bootstrap_top_box_lin)
                & (top[:, 10:13].norm(dim=-1) <= self._ss_bootstrap_top_box_ang)
            )
            self._ss_bootstrap_top_balance_count = torch.where(
                top_bootstrap_balanced,
                self._ss_bootstrap_top_balance_count + 1,
                torch.where(
                    pre_stack,
                    torch.zeros_like(self._ss_bootstrap_top_balance_count),
                    self._ss_bootstrap_top_balance_count,
                ),
            )
        if self._ss_stage_hold_steps > 0:
            stage_candidate = (
                pre_stack
                & stage_ready
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
            stage_valid = stage_ready & (stage_dist <= 1.5 * self._ss_stage_tol)
            lost_stage = pre_stack & self._ss_staged & (~stage_valid)
            if bool(lost_stage.any()):
                ids = torch.nonzero(lost_stage, as_tuple=False).squeeze(-1)
                self._ss_staged[ids] = False
                self._ss_stage_count[ids] = 0
                self._mscale[top_rows[ids]] = 1.0
                self._compute_observations()
        else:
            newly_staged = (
                pre_stack & (~self._ss_staged) & (stage_dist <= self._ss_stage_tol)
            )
        if bool(newly_staged.any()):
            ids = torch.nonzero(newly_staged, as_tuple=False).squeeze(-1)
            self._ss_staged[ids] = True
            self._ss_dbg_staged_step[ids] = self.progress_buf[ids]
            if self._ss_stage_force_zero:
                self._mscale[top_rows[ids]] = 0.0
            self._compute_observations()

        if self._ss_shared_goal_carry:
            top_waiting = (
                pre_stack
                & self._ss_staged
                & stage_ready
                & (stage_dist <= 1.5 * self._ss_stage_tol)
            )
            self._ss_top_wait_steps = torch.where(
                top_waiting,
                self._ss_top_wait_steps + 1,
                torch.zeros_like(self._ss_top_wait_steps),
            )
        else:
            self._ss_top_wait_steps.zero_()

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
            self._compute_observations()

        clear = self._ss_phase == self.CLEAR
        base_still_supported = (
            (xy_err <= self._ss_clear_xy_tol)
            & (z_err <= self._ss_clear_z_tol)
            & (base[:, 7:10].norm(dim=-1) <= self._ss_clear_stable_lin)
            & (base[:, 10:13].norm(dim=-1) <= self._ss_clear_stable_ang)
        )
        retreat_progress_done = (
            self._arc_root[base_rows] >= self._ss_clear_arc_dist
            if self._ss_clear_arc_dist > 0.0
            else root_dist >= self._ss_body_clear
        )
        if self._ss_clear_stop_on_stack:
            base_roots = self.humanoid_rows(
                self._humanoid_root_states
            )[base_rows]
            endpoint_error = (
                base_roots[:, 0:2] - self._ss_retreat_goal[:, 0:2]
            ).norm(dim=-1)
            just_arrived = (
                clear
                & retreat_progress_done
                & (endpoint_error <= 0.12)
                & (~self._ss_retreat_arrived)
            )
            if bool(just_arrived.any()):
                ids = torch.nonzero(just_arrived, as_tuple=False).squeeze(-1)
                self._ss_retreat_arrived[ids] = True
                self._hold_steer(base_rows[ids])
                self._compute_observations()
            retreat_done = self._ss_retreat_arrived
        else:
            retreat_done = retreat_progress_done
        if self._ss_stop_hold_steps > 0:
            remaining = (
                self._s_end[base_rows] - self._arc_root[base_rows]
            ).clamp(min=0.0)
            decel = (
                clear
                & (remaining <= self._ss_stop_decel_dist)
            )
            stop_stable = self._base_stop_features(base_rows)[-1]
            stop_candidate = (
                decel & clear_now & retreat_done & stop_stable
            )
            self._ss_stop_count = torch.where(
                stop_candidate,
                self._ss_stop_count + 1,
                torch.zeros_like(self._ss_stop_count),
            )
        else:
            self._ss_stop_count.zero_()
        ready_to_stack = clear & clear_now & retreat_done
        if self._ss_clear_hard_gate:
            ready_to_stack &= base_still_supported
        if self._ss_debug_require_top_balanced:
            top_roots = self.humanoid_rows(self._humanoid_root_states)[top_rows]
            top_balanced = (
                (top_roots[:, 7:10].norm(dim=-1) <= self._ss_stage_stable_lin)
                & (top_roots[:, 10:13].norm(dim=-1) <= self._ss_stage_stable_ang)
                & (
                    self._upright_score(top_roots[:, 3:7])
                    >= math.cos(math.radians(15.0))
                )
            )
            ready_to_stack &= top_balanced
        if self._ss_require_staged:
            ready_to_stack &= (
                self._ss_staged
                & stage_ready
                & (stage_dist <= 1.5 * self._ss_stage_tol)
            )
        # The strict multi-step stop gate is important for normal evaluation,
        # but stochastic PPO rollouts may never satisfy it long enough to seed
        # the late-phase curriculum.  Capture a physically valid one-step stop
        # without changing the real CLEAR->STACK transition condition.
        if (
            self._ss_bootstrap_capture_stable
            and self._ss_bootstrap_frac > 0.0
            and self._ss_stop_hold_steps > 0
        ):
            capture_ready = ready_to_stack & stop_stable
            if bool(capture_ready.any()):
                ids = torch.nonzero(capture_ready, as_tuple=False).squeeze(-1)
                self._capture_stack_bootstrap(ids)
        if self._ss_stop_hold_steps > 0:
            ready_to_stack &= self._ss_stop_count >= self._ss_stop_hold_steps
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
            if self._ss_top_commit_goal:
                self._commit_top_goal(ids)
            self._capture_stack_bootstrap(ids)
            trace_ids = self._claim_stack_trace(ids)
            self._record_stack_trace(trace_ids, -1)
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
            if not self._ss_debug_keep_wait:
                if self._ss_top_commit_goal:
                    self._box_tar_pos[rows] = self._ss_top_goal[ids]
                else:
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
            if not self._ss_debug_keep_wait:
                self._reset_steer(rows)
                self._mscale[rows] = self._ss_top_scale
            self._compute_observations()
            self._record_stack_trace(trace_ids, 0)

        self._ss_clear_age = torch.where(
            self._ss_phase == self.CLEAR,
            self._ss_clear_age + 1,
            torch.zeros_like(self._ss_clear_age),
        )

        active = self._ss_phase == self.STACK
        if bool(active.any()):
            ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            rows = top_rows[ids]
            if not self._ss_debug_keep_wait and not self._ss_top_commit_goal:
                follow = torch.ones(len(ids), dtype=torch.bool, device=self.device)
                if self._ss_bootstrap_keep_wait:
                    follow &= ~self._ss_bootstrap_active[ids]
                follow_ids = ids[follow]
                follow_rows = top_rows[follow_ids]
                desired = self._box_tar_pos[follow_rows].clone()
                desired[:, 0:2] = base[follow_ids, 0:2]
                base_size = self._box_lib._box_size[base_rows[follow_ids]]
                top_size = self._box_lib._box_size[follow_rows]
                desired[:, 2] = base[follow_ids, 2] + 0.5 * (
                    base_size[:, 2] + top_size[:, 2]
                )
                self._box_tar_pos[follow_rows] = desired
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
            parallel_error_deg = self._box_parallel_error_deg(
                base[ids, 3:7], top[ids, 3:7]
            )
            top_parallel_error_deg[ids] = parallel_error_deg
            # Quaternion arithmetic can report 15.00003 for an exact 15-degree
            # rotation, so keep the configured boundary inclusive.
            parallel_ok = (
                parallel_error_deg <= self._ss_top_parallel_deg + 1e-3
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
                & parallel_ok
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
            self._compute_observations()

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
        self.extras["stack_shared_path_min"] = self._ss_shared_path_min
        self.extras["stack_top_parallel_error_deg"] = top_parallel_error_deg
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
        self.extras["tb/stack_phase/bootstrap_fraction"] = (
            self._ss_bootstrap_active.float()
        )
        self.extras["tb/stack_state/top_parallel_error_deg"] = (
            top_parallel_error_deg
        )
        self.extras["tb/stack_state/shared_path_min"] = self._ss_shared_path_min
        self.extras["tb/stack_state/shared_path_safe"] = (
            self._ss_shared_path_min >= self._ss_shared_path_clearance
        ).float()
        # Separate an empty bootstrap bank from a valid bank that simply was
        # not sampled on the current reset. Occupancy is not the reset draw rate.
        self.extras["tb/stack_phase/bootstrap_valid_fraction"] = (
            getattr(
                self,
                "_ss_bootstrap_valid",
                torch.zeros_like(self._ss_bootstrap_active),
            ).float()
        )
        self.extras["tb/stack_phase/bootstrap_target_fraction"] = (
            torch.ones_like(self._ss_bootstrap_active, dtype=torch.float)
            * self._ss_bootstrap_frac * (1.0 - self._ss_rehearsal_frac)
        )
        self.extras["tb/stack_phase/bootstrap_draw_probability"] = (
            torch.ones_like(self._ss_bootstrap_active, dtype=torch.float)
            * self._ss_bootstrap_frac
        )
        bootstrap_reset_fraction = self._ss_bootstrap_reset_draw_count / torch.clamp(
            self._ss_total_reset_count, min=1.0
        )
        bootstrap_eligible_draw_fraction = (
            self._ss_bootstrap_reset_draw_count
            / torch.clamp(self._ss_bootstrap_eligible_reset_count, min=1.0)
        )
        self.extras["tb/stack_phase/bootstrap_reset_fraction"] = (
            torch.ones_like(self._ss_bootstrap_active, dtype=torch.float)
            * bootstrap_reset_fraction
        )
        self.extras["tb/stack_phase/bootstrap_eligible_draw_fraction"] = (
            torch.ones_like(self._ss_bootstrap_active, dtype=torch.float)
            * bootstrap_eligible_draw_fraction
        )
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
        self._record_active_stack_trace()
        return
