#!/bin/bash
# Continue ms18 for 500 iterations with the real placed box hidden only from
# carry observation and a stationary virtual box placed at the rear endpoint.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=ms38_ms18init_seqrewardmask_side135_dynmask_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms40_ms18init_seqrewardmask_side135_virtualrear_steerunfreeze_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_ITERS=${PILOT_ITERS:-500}
export MS_SAVE_LATEST=100
export MS_SAVE_ARCHIVE=100
export MA_GPU=${MA_GPU:-7}

# Keep trainability identical to ms39 so the CLEAR carry representation is
# the only comparison axis: zero/masked carry versus a rear virtual box.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_VIRTUAL_RETREAT_BOX=1
export STACK_VIRTUAL_RETREAT_REAR_BOX=1
export STACK_CARRY_FOOT_GATE=1
export STACK_FOOT_CLEAR=0.20
export STACK_FOOT_BOX_W=${PILOT_FOOT_BOX_W:-0.50}
export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
