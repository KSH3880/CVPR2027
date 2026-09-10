#!/bin/bash
# Reward-continuity pilot: preserve a small completed-CARRY floor after the
# phase transition while retaining ms41's signed CLEAR objective.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms39_ms18init_seqrewardmask_side135_dynmask_steerunfreeze_foot050_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_MB=${PILOT_MB:-16384}
export MS_ITERS=${PILOT_ITERS:-500}
export MS_SAVE_LATEST=${PILOT_SAVE_EVERY:-100}
export MS_SAVE_ARCHIVE=${PILOT_SAVE_EVERY:-100}
export MA_GPU=${MA_GPU:-7}

# Preserve ms41's trainable/frozen parameter split.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1

# Keep carry observations zero after release without masking the frozen
# Transformer's carry-token positions.
export STACK_ZERO_CARRY_OBS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_VIRTUAL_RETREAT_BOX=0
export STACK_VIRTUAL_RETREAT_REAR_BOX=0

# Keep a modest CARRY-completion floor across later phases.  RELEASE/CLEAR
# completion constants remain zero so an idle CLEAR state is not rewarded.
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=${PILOT_CARRY_DONE_REWARD:-0.50}
export STACK_RELEASE_DONE_REWARD=${PILOT_RELEASE_DONE_REWARD:-0.00}
export STACK_CLEAR_DONE_REWARD=${PILOT_CLEAR_DONE_REWARD:-0.00}
export STACK_NEGATIVE_CLEAR_REWARD=1
export STACK_CLEAR_MOVE_W=${PILOT_MOVE_W:-1.0}
export STACK_CLEAR_HAND_PEN_W=${PILOT_HAND_PEN_W:-0.5}
export STACK_CLEAR_STALL_PEN_W=${PILOT_STALL_PEN_W:-0.5}
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
