#!/bin/bash
# ms69: preserve ms67's rear-biased CLEAR contract, extend retreat to 2--3 m,
# and strengthen Base's STACK hold reward without ms68's wait-stage bypass.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms69_ms18e9000_fade3_angle120_retreat2to3_hold05_3000_s0}

# Local pilot profile.
export PILOT_GPU=1
export PILOT_ENVS=${PILOT_ENVS:-1024}
export PILOT_ITERS=${PILOT_ITERS:-1000}

# Sample 0..60 degrees away from the strict outward ray. In box-relative
# coordinates this is the same rear-biased 120..180 degree range as ms67.
export PILOT_RETREAT_SIDE_DEG=60
export PILOT_RETREAT_SIDE_BINS=1
export PILOT_RETREAT_RANDOM=1
export PILOT_RETREAT_DIST=3.0
export PILOT_RETREAT_DIST_MIN=2.0
export PILOT_CLEAR_ARC_DIST=0.60
# Explicitly retain ms67's mandatory Top safety-gate contract.
export STACK_CLEAR_BYPASS_STAGE=0

# Fade both released-carrier carry windows to zero in 3 policy steps at 30 Hz
# (about 0.10 s, or 6 simulator frames at 60 Hz).
export PILOT_VIRTUAL_RETREAT_BOX=0
export PILOT_VIRTUAL_RETREAT_REAR_BOX=0
export PILOT_ZERO_CARRY_OBS=0
export PILOT_DYNAMIC_CARRY_MASK=0
export PILOT_CARRY_OBS_ZERO_FADE=1
export PILOT_CARRY_OBS_ZERO_FADE_STEPS=3

# Keep ms67's single CLEAR phase and classic steering contract.
export PILOT_NEGATIVE_CLEAR_REWARD=0
export PILOT_CLEAR_CLASSIC_STEER=1
export PILOT_CLEAR_CLASSIC_STOP_REWARD=1
export PILOT_CLEAR_HEADING_PROGRESS_W=2.0
export PILOT_CLEAR_FORWARD_GATE=1
export PILOT_CLEAR_MOVE_W=0.0
export PILOT_CLEAR_STALL_PEN_W=0.10
export PILOT_CLEAR_PATH_PEN_W=0.0
export PILOT_STOP_DECEL_DIST=0.40
export PILOT_STOP_HOLD_STEPS=5
export PILOT_STOP_LIN=0.20
export PILOT_STOP_ANG=1.00
export PILOT_STOP_UPRIGHT_DEG=25.0
export PILOT_STOP_CONTACT_FORCE=0.50
export PILOT_TOP_WAIT_REWARD_W=0.10
export PILOT_BASE_HOLD_REWARD_W=0.10
export PILOT_BASE_REGRASP_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_ms52_shared_goal_safe_local.sh" "$TAG"
