#!/bin/bash
# MS82 stop-v2: continuous solo stop reward, anchor-consistent goal, carry-safe FT.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=ms18_carry_steer50_s0
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_23-12-39-56/nn/Humanoid_00012000.pth"
GPU=${MA_GPU:-6}
if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU must be 6 or 7: $GPU" >&2
    exit 2
fi
if [[ ! -f "$ROOT/runs/queue/logs/$BASE_TAG.env" || ! -f "$INIT_CKPT" ]]; then
    echo "기준 sidecar 또는 epoch-12000 체크포인트 없음" >&2
    exit 1
fi

source "$ROOT/runs/queue/logs/$BASE_TAG.env"
export MS_TAG=${1:-ms83_ms18e12000_stopfix_steeronly_s0}
export MA_GPU="$GPU"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_TASK=HumanoidMACarrySteerStopMix

export MS_STOP_PROB=${MS_STOP_PROB:-0.5}
export MS_STOP_BRAKE_T=${MS_STOP_BRAKE_T:-0.8}
export MS_STOP_RESUME_T=${MS_STOP_RESUME_T:-0.8}
export MS_STOP_HOLD_MIN=${MS_STOP_HOLD_MIN:-0.5}
export MS_STOP_HOLD_MAX=${MS_STOP_HOLD_MAX:-2.0}
export MS_STOP_MARGIN=${MS_STOP_MARGIN:-0.8}
export MS_STOP_POST_BOX=${MS_STOP_POST_BOX:-0.8}
export MS_STOP_BRAKE_DIST=${MS_STOP_BRAKE_DIST:-0.6}
export MS_STOP_REWARD_W=${MS_STOP_REWARD_W:-1.0}
export MS_STOP_FIX_SOLO_REWARD=1
export MS_STOP_ANCHOR_GOAL=1
export MS_STOP_TB=1

# Protect MS18 carry modules from shared-residual drift; only steering may move.
export MA_FREEZE_NEW_CARRY=1 MA_ADAPTER_ONLY=0 MA_FINETUNE_STEER_ONLY=1
unset MA_FINETUNE_NEWCARRY_RESIDUAL
exec bash "$ROOT/scripts/masteer/train_local.sh" "$MS_TAG"
