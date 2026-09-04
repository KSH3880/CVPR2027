#!/bin/bash
# Preserve ms20's learned carry/release policy and adapt only steering/adapter
# parameters to the ms21 path-progress CLEAR objective.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms22_ms20init_pathclear_s0}

MS20_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms20_stackrel_gate_bonus3_s0/Humanoid_03-14-18-00/nn/Humanoid.pth"
if [ ! -f "$MS20_CKPT" ]; then
    echo "ms20 초기 체크포인트 없음: $MS20_CKPT" >&2
    exit 1
fi

# Load the complete ms20 epoch-9300 state.  Freeze new_carry only after loading
# it, so its value is ms20's learned release representation rather than ms18's.
export MA_INIT_CKPT="$MS20_CKPT"
export MA_FREEZE_NEW_CARRY=1

exec bash "$ROOT/scripts/masteer/train_stack_release_local.sh" "$TAG"
