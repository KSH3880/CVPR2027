#!/bin/bash
# Train the ms37 sequential task from ms18 epoch 9000 with a TokenHSI-style
# dynamic task-token mask. No token, layer, or observation dimension is added.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_TAG=ms37_ms18init_seqrewardmask_side135_s0_try2
CONFIG_ENV="$ROOT/runs/queue/logs/$CONFIG_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms38_ms18init_seqrewardmask_side135_dynmask_s0}

if [ ! -f "$CONFIG_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $CONFIG_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

# Reproduce every ms37 task/reward/path setting. The policy itself starts from
# ms18 epoch 9000; the only new experimental axis is post-CLEAR carry masking.
source "$CONFIG_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=1024
export MS_ITERS=3000
export MS_SAVE_LATEST=100
export MS_SAVE_ARCHIVE=100
export MA_GPU=${MA_GPU:-7}
export STACK_DYNAMIC_CARRY_MASK=1
export STACK_VIRTUAL_RETREAT_BOX=0
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
