#!/bin/bash
# ms70: keep one CLEAR phase while turning toward a rear-biased path, use direct
# yaw-error progress, fade released-box observations, walk forward, and stop.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms70_ms18e9000_yaw15_wait10_hold10_1000_s0}

# Local pilot profile. Do not alter the running ms66 wrapper in place.
export PILOT_GPU=1
export PILOT_ENVS=${PILOT_ENVS:-1024}
export PILOT_ITERS=${PILOT_ITERS:-1000}

# Sample 0..60 degrees away from the strict outward ray. In box-relative
# coordinates this is the requested rear-biased 120..180 degree range.
export PILOT_RETREAT_SIDE_DEG=60
export PILOT_RETREAT_SIDE_BINS=1
export PILOT_RETREAT_RANDOM=1
export PILOT_RETREAT_DIST=3.0
export PILOT_RETREAT_DIST_MIN=2.0
export PILOT_CLEAR_ARC_DIST=0.60

# If Base finishes CLEAR before Top reaches the safety gate, commit Top's
# target directly to the settled Base box instead of completing the obsolete
# wait-point detour.  Top still uses and waits at the gate when it arrives first.
export PILOT_CLEAR_BYPASS_STAGE=${PILOT_CLEAR_BYPASS_STAGE:-1}

# Fade both released-carrier carry windows to zero in 3 policy steps at 30 Hz
# (about 0.10 s, or 6 simulator frames at 60 Hz).
export PILOT_VIRTUAL_RETREAT_BOX=0
export PILOT_VIRTUAL_RETREAT_REAR_BOX=0
export PILOT_ZERO_CARRY_OBS=0
export PILOT_DYNAMIC_CARRY_MASK=0
export PILOT_CARRY_OBS_ZERO_FADE=1
export PILOT_CARRY_OBS_ZERO_FADE_STEPS=3

# A single CLEAR phase: turn progress, heading-gated classic path walking, a
# small post-turn stall penalty, then the existing endpoint stop objective.
export PILOT_NEGATIVE_CLEAR_REWARD=0
export PILOT_CLEAR_CLASSIC_STEER=1
export PILOT_CLEAR_CLASSIC_STOP_REWARD=1
export PILOT_CLEAR_HEADING_PROGRESS_W=${PILOT_CLEAR_HEADING_PROGRESS_W:-1.5}
export PILOT_CLEAR_YAW_PROGRESS=${PILOT_CLEAR_YAW_PROGRESS:-1}
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
export PILOT_TOP_WAIT_REWARD_W=${PILOT_TOP_WAIT_REWARD_W:-10.0}
export PILOT_BASE_HOLD_REWARD_W=${PILOT_BASE_HOLD_REWARD_W:-10.0}
export PILOT_BASE_REGRASP_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_ms52_shared_goal_safe_local.sh" "$TAG"
