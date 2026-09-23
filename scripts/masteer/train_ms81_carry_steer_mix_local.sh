#!/bin/bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=ms81_ms18e9000_cleargoal3_nobaseboxobs_3000_s0
GPU=${MA_GPU:-6}
if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU must be 6 or 7: $GPU" >&2
    exit 2
fi
if [[ ! -f "$ROOT/runs/queue/logs/$BASE_TAG.env" ]]; then
    echo "MS81 sidecar 없음" >&2
    exit 1
fi
source "$ROOT/runs/queue/logs/$BASE_TAG.env"
export MS_TAG=${1:-ms18_carry_steer50_s0}
export MA_GPU="$GPU"
export MS_TASK=HumanoidMACarrySteerMix
export MA_FREEZE_NEW_CARRY=1 MA_ADAPTER_ONLY=0
unset MA_FINETUNE_NEWCARRY_RESIDUAL MA_FINETUNE_STEER_ONLY
export MS_STEER_ONLY_DIST=3.0
exec bash "$ROOT/scripts/masteer/train_local.sh" "$MS_TAG"
