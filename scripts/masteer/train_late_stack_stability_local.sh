#!/bin/bash
# ms49 late-STACK stability curriculum. Reuse physical CLEAR->STACK states
# collected online, then replay a subset with both agents holding for 180 frames.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms48_ms18init_wait1_stack180_holddebug_1000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_10-13-32-32/nn/Humanoid_00009700.pth"
TAG=${1:-ms49_ms48e9700_latehold_boot50_stab100_1000_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_MB=${PILOT_MB:-16384}
export MS_ITERS=${PILOT_ITERS:-1000}
export MS_SAVE_LATEST=${PILOT_SAVE_EVERY:-100}
export MS_SAVE_ARCHIVE=${PILOT_SAVE_EVERY:-100}
export MA_GPU=${MA_GPU:-7}
export MS_EPISODE_LENGTH=780

# Keep the ms48 actor, 340-D observation, token layout, and freeze contract.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_ZERO_CARRY_OBS=1

# Normal sequential episodes first collect authentic transition states. Once an
# env has one, half of its later resets replay that state as a 180-frame hold.
export STACK_REHEARSAL_FRAC=0.30
export STACK_BOOTSTRAP_FRAC=0.50
export STACK_BOOTSTRAP_KEEP_WAIT=1
export STACK_BOOTSTRAP_EVAL=0
export STACK_HUMANOID_STABILITY_W=1.00
export STACK_PRE_STEPS=600
export STACK_PHASE_STEPS=180

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
