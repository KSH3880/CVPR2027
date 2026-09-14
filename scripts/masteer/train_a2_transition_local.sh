#!/bin/bash
# Continue ms50r1 with Juan-style A2 staging and full-scale STACK restart.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms50r1_ms18init_boot100_hold_stab100_1000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
DEFAULT_INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_11-16-27-47/nn/Humanoid_00009300.pth"
INIT_CKPT=${INIT_CKPT_OVERRIDE:-$DEFAULT_INIT_CKPT}
TAG=${1:-ms51_ms50r1e9300_a2transition_1000_s0}

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

# Keep A2's WAIT command in the pretrained carry regime: farther staging at
# the current mean hand height, no abrupt mscale zero, then full-scale restart.
export STACK_STAGE_DIST=1.5
export STACK_STAGE_USE_HAND_Z=1
export STACK_STAGE_FORCE_ZERO=0
export STACK_TOP_SCALE=1.0
export STACK_BOOTSTRAP_KEEP_WAIT=0

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
