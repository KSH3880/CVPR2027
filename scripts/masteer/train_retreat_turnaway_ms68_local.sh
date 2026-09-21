#!/bin/bash
# ms68: keep one CLEAR phase while turning toward a rear-biased path, fading
# released-box observations in three policy steps, walking forward, and stopping.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms68_ms18e9000_fade3_angle120_retreat2to3_3000_s0}
GPU=${MA_GPU:-7}
if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU must be 6 or 7: $GPU" >&2
    exit 2
fi
export MA_GPU="$GPU"


# Server profile. Select an allowed GPU at launch with MA_GPU=6 or MA_GPU=7.
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
export PILOT_CLEAR_ROUTE_AROUND=0

# If Base finishes CLEAR before Top reaches the safety gate, commit Top's
# target directly to the settled Base box instead of completing the obsolete
# wait-point detour.  Top still uses and waits at the gate when it arrives first.
export PILOT_CLEAR_BYPASS_STAGE=1

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
export PILOT_CLEAR_HEADING_PROGRESS_W=2.0
export PILOT_CLEAR_FACING_W=0.0
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
export PILOT_STOP_REWARD_W=1.0
export PILOT_TOP_WAIT_REWARD_W=0.50
export PILOT_BASE_HOLD_REWARD_W=0.50
export PILOT_BASE_REGRASP_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_ms52_shared_goal_safe_local.sh" "$TAG"
