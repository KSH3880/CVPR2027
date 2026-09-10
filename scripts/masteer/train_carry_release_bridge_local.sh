#!/bin/bash
# ms44: keep ms43's RELEASE -> CLEAR continuity and add an exact
# CARRY -> RELEASE reward bridge. Model, observations, freeze split, token
# masks, and all controller/gate settings remain identical to ms43.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms43_ms18init_negclear_carry050_release100_stall150_zeroobs_nomask_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms44_ms18init_carryreleasebridge_rel100_stall150_3000_s0}

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

# Keep ms43's proven trainable/frozen split. Do not unfreeze the carry
# tokenizer and do not reintroduce the post-CLEAR dynamic attention mask.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_ZERO_CARRY_OBS=1

# h=0 starts at the native CARRY reward; h=1 reaches carry_done + RELEASE
# local reward. CLEAR still starts at carry_done + release_done = 1.5, and
# the post-grace 1.5 stall penalty still cancels a stationary solution.
export STACK_RELEASE_CARRY_BRIDGE=1
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=0.50
export STACK_RELEASE_DONE_REWARD=1.00
export STACK_CLEAR_DONE_REWARD=0.00
export STACK_CLEAR_STALL_PEN_W=1.50
export STACK_CLEAR_GRACE_STEPS=5

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
