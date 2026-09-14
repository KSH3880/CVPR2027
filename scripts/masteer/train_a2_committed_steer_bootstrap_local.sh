#!/bin/bash
# A1 steering-only CLEAR + Juan-style A2 wait/committed STACK continuation.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms52_ms18init_a2trans_w2s_steer100_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/$BASE_TAG/Humanoid_11-19-44-25/nn/Humanoid_00012000.pth"
FREEZE_A2_CARRY_MODE=${PILOT_FREEZE_A2_CARRY:-0}
case "$FREEZE_A2_CARRY_MODE" in
    0) DEFAULT_TAG=ms54_ms52e12000_juanwait_commit_steerclear_boot50_3000_s0 ;;
    1) DEFAULT_TAG=ms55_ms52e12000_freezea2carry_juanwait_commit_steerclear_boot50_3000_s0 ;;
    *) echo "PILOT_FREEZE_A2_CARRY는 0 또는 1이어야 합니다." >&2; exit 2 ;;
esac
TAG=${1:-$DEFAULT_TAG}

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

# Default Juan freeze contract updates new_carry plus the existing shared
# action residual. The ablation freezes new_carry while leaving the steering
# extra tokenizer and shared residual trainable.
export MA_NOFREEZE=0
if [ "$FREEZE_A2_CARRY_MODE" = 1 ]; then
    export MA_ADAPTER_ONLY=0
    export MA_FREEZE_NEW_CARRY=1
    export MA_FINETUNE_NEWCARRY_RESIDUAL=0
else
    export MA_ADAPTER_ONLY=0
    export MA_FREEZE_NEW_CARRY=0
    export MA_FINETUNE_NEWCARRY_RESIDUAL=1
fi
export MA_FREEZE_INPUT_RMS=1
export MA_TOKEN=mask

# A1 CLEAR is genuinely steering-only. Both zero carry observation windows are
# excluded from Transformer attention instead of becoming normalized constants.
export STACK_VIRTUAL_RETREAT_BOX=0
export STACK_VIRTUAL_RETREAT_REAR_BOX=0
export STACK_ZERO_CARRY_OBS=1
export STACK_DYNAMIC_CARRY_MASK=1

# A2 waits at its own reset box. Once A1 is clear, commit one top pose from the
# settled physical base and regenerate normal steering to that same carry goal.
export STACK_TOP_WAIT_AT_START=1
export STACK_STAGE_FORCE_ZERO=0
export STACK_TOP_COMMIT_GOAL=1
export STACK_TOP_CARRY_TARGET_ONLY=0
unset STACK_DEBUG_TOP_CARRY_TARGET_ONLY
export STACK_TOP_DIRECT_CARRY_REWARD=1
export STACK_TOP_SCALE=1.0
export STACK_BOOTSTRAP_KEEP_WAIT=0

# Admit a late-CLEAR snapshot only after A2 and its waiting box remain balanced
# for 12 consecutive frames. A2 has not picked the box up in this wait design,
# so grasp is deliberately not part of the snapshot predicate.
export STACK_BOOTSTRAP_FRAC=0.50
export STACK_BOOTSTRAP_CAPTURE_STABLE=1
export STACK_BOOTSTRAP_TOP_BALANCE_STEPS=12
export STACK_BOOTSTRAP_TOP_ROOT_LIN=0.10
export STACK_BOOTSTRAP_TOP_ROOT_ANG=0.50
export STACK_BOOTSTRAP_TOP_UPRIGHT_DEG=15.0
export STACK_BOOTSTRAP_TOP_BOX_LIN=0.15
export STACK_BOOTSTRAP_TOP_BOX_ANG=0.40
export STACK_BOOTSTRAP_TOP_REQUIRE_GRASP=0
export STACK_BOOTSTRAP_EVAL=0
export STACK_REHEARSAL_FRAC=0.30
export STACK_HUMANOID_STABILITY_W=1.00

# Training metrics are intentionally disabled. train_local.sh now preserves
# this unset instead of silently recreating a multi-GiB default npy.
unset MS_METRICS MA_METRICS
export MS_GRADCHK=0

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
