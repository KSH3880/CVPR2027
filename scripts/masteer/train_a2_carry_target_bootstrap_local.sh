#!/bin/bash
# ms53: continue ms52 with A2's post-CLEAR steering values zero-padded and the
# native Juan carry objective pointed directly at the stacked-box target.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms52_ms18init_a2trans_w2s_steer100_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_11-19-44-25/nn/Humanoid_00012000.pth"
TAG=${1:-ms53_ms52e12000_a2carryonly_boot50_3000_s0}

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
export MS_EPISODE_LENGTH=780

# Preserve ms52's shared-policy/freeze contract and A1 zero-padding behavior.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_ZERO_CARRY_OBS=1
export STACK_DYNAMIC_CARRY_MASK=0

# A2 keeps both learned carry tokens and their attention slots. Only its raw
# steering slice is zero after entering STACK, and reward uses the live stack
# carry target rather than the freshly generated steering path.
export STACK_TOP_CARRY_TARGET_ONLY=1
unset STACK_DEBUG_TOP_CARRY_TARGET_ONLY
export STACK_TOP_DIRECT_CARRY_REWARD=1

# Collect genuine, one-step-stable late-CLEAR snapshots online without relaxing
# ms52's normal 15-step phase gate, then replay half of eligible resets with A2
# immediately attempting the direct carry and A1 held in place.
export STACK_REHEARSAL_FRAC=0.30
export STACK_BOOTSTRAP_FRAC=0.50
export STACK_BOOTSTRAP_KEEP_WAIT=0
export STACK_BOOTSTRAP_CAPTURE_STABLE=1
export STACK_BOOTSTRAP_EVAL=0
export STACK_HUMANOID_STABILITY_W=1.00

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
