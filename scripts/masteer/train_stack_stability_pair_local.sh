#!/bin/bash
# Compare zero steering against a small continuous loop in late STACK.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
MODE=${1:-}
TAG=${2:-}
BASE_ENV="$ROOT/runs/queue/logs/ms48_ms18init_wait1_stack180_holddebug_1000_s0.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"

case "$MODE" in
    hold)
        LOOP_RADIUS=0.0
        ;;
    loop)
        LOOP_RADIUS=0.4
        ;;
    *)
        echo "사용법: train_stack_stability_pair_local.sh <hold|loop> <tag>" >&2
        exit 2
        ;;
esac

if [ -z "$TAG" ] || [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "tag 또는 기준 파일 없음: $BASE_ENV / $INIT_CKPT" >&2
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

export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1

export STACK_REHEARSAL_FRAC=0.10
export STACK_BOOTSTRAP_FRAC=1.00
export STACK_BOOTSTRAP_KEEP_WAIT=1
export STACK_BOOTSTRAP_EVAL=0
export STACK_BOOTSTRAP_LOOP_RADIUS="$LOOP_RADIUS"
export STACK_BOOTSTRAP_LOOP_SCALE=0.25
export STACK_BOOTSTRAP_LOOP_TRACK_W=1.00
export STACK_HUMANOID_STABILITY_W=1.00
export STACK_PRE_STEPS=600
export STACK_PHASE_STEPS=180

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
