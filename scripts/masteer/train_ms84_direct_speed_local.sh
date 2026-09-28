#!/bin/bash
# MS84: abrupt M<->0 command switching with speed tracking only.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=ms18_carry_steer50_s0
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_23-12-39-56/nn/Humanoid_00012000.pth"

# Capture caller overrides before sourcing the MS18 replay sidecar.
GPU=${MA_GPU:-6}
ITERS=${MS_ITERS:-3000}
ENVS=${MS_ENVS:-1024}
CP=${MS_CP:-4}
EPISODE_LENGTH=${MS_EPISODE_LENGTH:-780}
TAG=${1:-ms84_ms18e12000_directspeed_steeronly_e2048_s0}

if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU must be 6 or 7: $GPU" >&2
    exit 2
fi
if [[ ! -f "$ROOT/runs/queue/logs/$BASE_TAG.env" || ! -f "$INIT_CKPT" ]]; then
    echo "기준 sidecar 또는 epoch-12000 체크포인트 없음" >&2
    exit 1
fi

source "$ROOT/runs/queue/logs/$BASE_TAG.env"
export MS_TAG="$TAG"
export MA_GPU="$GPU"
export MS_ITERS="$ITERS" MS_ENVS="$ENVS" MS_CP="$CP"
export MS_EPISODE_LENGTH="$EPISODE_LENGTH"
export MS_SAVE_LATEST=100 MS_SAVE_ARCHIVE=100
export MA_INIT_CKPT="$INIT_CKPT"
export MS_TASK=HumanoidMACarrySteerStopMix

# Binary time switch only: no cosine brake/resume and no additive stop quality.
export MS_STOP_PROB=${MS_STOP_PROB:-0.5}
export MS_STOP_BRAKE_T=0.8
export MS_STOP_RESUME_T=0.8
export MS_STOP_HOLD_MIN=${MS_STOP_HOLD_MIN:-0.5}
export MS_STOP_HOLD_MAX=${MS_STOP_HOLD_MAX:-2.0}
export MS_STOP_MARGIN=${MS_STOP_MARGIN:-0.8}
export MS_STOP_POST_BOX=${MS_STOP_POST_BOX:-0.8}
export MS_STOP_BRAKE_DIST=0.0
export MS_STOP_REWARD_W=0.0
export MS_STOP_DIRECT_SPEED=1
export MS_STOP_FIX_SOLO_REWARD=1
export MS_STOP_ANCHOR_GOAL=0
export MS_STOP_TB=1

# Preserve the exact-tracking peak and add only a dense large-overspeed tail.
export MS_VEL_W=1.0
export MS_SPEED_OVER_W=${MS_SPEED_OVER_W:-1.0}
export MS_SPEED_OVER_TOL=${MS_SPEED_OVER_TOL:-0.05}
export MS_SPEED_OVER_BETA=${MS_SPEED_OVER_BETA:-0.20}
export MS_SPEED_TB=1

# Protect MS18 carry/backbone; train only the steering tokenizer.
export MA_FREEZE_NEW_CARRY=1 MA_ADAPTER_ONLY=0 MA_FINETUNE_STEER_ONLY=1
unset MA_FINETUNE_NEWCARRY_RESIDUAL

exec bash "$ROOT/scripts/masteer/train_local.sh" "$MS_TAG"
