#!/bin/bash
# RELEASE-to-CLEAR continuity pilot: retain completed RELEASE value in CLEAR
# and increase the post-grace stall cost so standing still is not profitable.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms39_ms18init_seqrewardmask_side135_dynmask_steerunfreeze_foot050_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms43_ms18init_negclear_carry050_release100_stall150_zeroobs_nomask_3000_s0}

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

# Preserve ms42's trainable/frozen parameter split.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1

# Keep carry observations zero after CLEAR without masking carry-token
# positions in the frozen Transformer.
export STACK_ZERO_CARRY_OBS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_VIRTUAL_RETREAT_BOX=0
export STACK_VIRTUAL_RETREAT_REAR_BOX=0

# Match the successful RELEASE boundary at 1.5 reward during CLEAR grace:
# carry 0.5 + release 1.0. After grace, a stationary CLEAR receives stall
# -1.5, cancelling the completion floor before other penalties.
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=${PILOT_CARRY_DONE_REWARD:-0.50}
export STACK_RELEASE_DONE_REWARD=${PILOT_RELEASE_DONE_REWARD:-1.00}
export STACK_CLEAR_DONE_REWARD=${PILOT_CLEAR_DONE_REWARD:-0.00}
export STACK_NEGATIVE_CLEAR_REWARD=1
export STACK_CLEAR_MOVE_W=${PILOT_MOVE_W:-1.0}
export STACK_CLEAR_HAND_PEN_W=${PILOT_HAND_PEN_W:-0.5}
export STACK_CLEAR_STALL_PEN_W=${PILOT_STALL_PEN_W:-1.5}
export STACK_CLEAR_REVERSE_PEN_W=${PILOT_REVERSE_PEN_W:-0.5}
export STACK_CLEAR_MOVE_MIN_FRAC=${PILOT_MOVE_MIN_FRAC:-0.20}
export STACK_CLEAR_GRACE_STEPS=${PILOT_GRACE_STEPS:-5}

export STACK_REHEARSAL_FRAC=${PILOT_REHEARSAL_FRAC:-0.50}
export STACK_CARRY_FOOT_GATE=1
export STACK_FOOT_CLEAR=0.20
export STACK_FOOT_BOX_W=${PILOT_FOOT_BOX_W:-0.10}
export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
