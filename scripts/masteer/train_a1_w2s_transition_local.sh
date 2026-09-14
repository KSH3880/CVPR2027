#!/bin/bash
# Train A1 CLEAR->DECEL->stand and the retained A2 transition from ms18.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms50r1_ms18init_boot100_hold_stab100_1000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms52_ms18init_a2trans_w2s_steer100_3000_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_MB=${PILOT_MB:-16384}
export MS_ITERS=${PILOT_ITERS:-3000}
export MS_SAVE_LATEST=${PILOT_SAVE_EVERY:-100}
export MS_SAVE_ARCHIVE=${PILOT_SAVE_EVERY:-100}
export MA_GPU=${MA_GPU:-7}

# Keep the A2 transition contract from train_a2_transition_local.sh.
export STACK_STAGE_DIST=1.5
export STACK_STAGE_USE_HAND_Z=1
export STACK_STAGE_FORCE_ZERO=0
export STACK_TOP_SCALE=1.0

# Strengthen A1 retreat steering, then route the endpoint into deceleration
# and require about 0.33 s of physical double-support stability before STACK.
export STACK_CLEAR_STEER_W=1.0
export STACK_STOP_DECEL_DIST=0.30
export STACK_STOP_HOLD_STEPS=${PILOT_STOP_HOLD_STEPS:-10}
export STACK_STOP_LIN=0.10
export STACK_STOP_ANG=0.50
export STACK_STOP_UPRIGHT_DEG=15.0
export STACK_STOP_CONTACT_FORCE=1.0
export STACK_STOP_REWARD_W=1.0

# Pay terminal success only when the two box face frames are parallel. Quarter
# turns are equivalent for cube/square boxes; a 45-degree diamond is rejected.
export STACK_TOP_PARALLEL_DEG=${PILOT_TOP_PARALLEL_DEG:-15.0}

# Full sequential episodes must traverse and learn the new stopping phase.
export STACK_BOOTSTRAP_FRAC=0.0
export STACK_BOOTSTRAP_KEEP_WAIT=0

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
