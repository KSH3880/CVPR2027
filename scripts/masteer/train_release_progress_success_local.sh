#!/bin/bash
# ms45: remove ms44's live CARRY reward from RELEASE. Reward actual hand
# separation progress, penalize lingering contact after a short grace period,
# and pay a large one-shot bonus to both agents only after strict final stacking.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms44_ms18init_carryreleasebridge_rel100_stall150_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms45_ms18init_relprogress050_hold050_success10_3000_s0}

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

# Preserve ms44's model, observation, controller, rehearsal, and freeze split.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_ZERO_CARRY_OBS=1

# RELEASE is phase-isolated: no live native-CARRY reward. The steady reward is
# 0.5 + 0.5*h + clear*(0.3*support + 0.2*stable), with signed hand-separation
# progress and a post-grace penalty for staying attached.
export STACK_RELEASE_CARRY_BRIDGE=0
export STACK_RELEASE_PROGRESS_W=0.50
export STACK_RELEASE_HOLD_PEN_W=0.50
export STACK_RELEASE_HOLD_GRACE_STEPS=5

# Strict top-on-base success already requires 20 stable frames. Pay +10 once
# to both agents on that rising edge; keep the existing +0.5/step success floor.
export STACK_SUCCESS_BONUS=10.0
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=0.50
export STACK_RELEASE_DONE_REWARD=1.00
export STACK_CLEAR_DONE_REWARD=0.00
export STACK_CLEAR_STALL_PEN_W=1.50
export STACK_CLEAR_GRACE_STEPS=5

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
