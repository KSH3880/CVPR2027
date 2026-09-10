#!/bin/bash
# ms46 bootstrap pilot: preserve ms18 carry, route the released base carrier
# around the placed box, then reward top-box arrival and physical release.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms45_ms18init_relprogress050_hold050_success10_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms46_ms18init_curveclear_stackfirst_1000_s0}

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

# Preserve the ms45 model/observation layout. Increase ordinary carry rehearsal
# and the completed-CARRY floor to reduce pre-STACK forgetting during the pilot.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_ZERO_CARRY_OBS=1
export STACK_REHEARSAL_FRAC=0.65
export STACK_CARRY_DONE_REWARD=1.00

# Keep base release shaping, but make CLEAR a slow tangent route around the box.
# Small contact is soft: no foot gate, a reduced penalty, and 0.35 m drop margin.
export STACK_RELEASE_CARRY_BRIDGE=0
export STACK_RELEASE_PROGRESS_W=0.50
export STACK_RELEASE_HOLD_PEN_W=0.50
export STACK_RELEASE_HOLD_GRACE_STEPS=5
export STACK_CLEAR_ROUTE_AROUND=1
export STACK_CLEAR_ROUTE_MARGIN=0.25
export STACK_CLEAR_STOP_ON_STACK=1
export STACK_RETREAT_DIST=0.75
export STACK_RETREAT_SCALE=0.25
export STACK_CLEAR_ARC_DIST=0.60
export STACK_CLEAR_GRACE_STEPS=20
export STACK_CLEAR_STALL_PEN_W=0.75
export STACK_ENTRY_FOOT_GATE=0
export STACK_RELEASE_FOOT_GATE=0
export STACK_FOOT_BOX_W=0.05
export STACK_DROP_XY=0.35

# Bootstrap stacking: slow top approach, one-shot above-box credit, then reward
# actual hand clearance and a short physically stable placement.
export STACK_TOP_SCALE=0.35
export STACK_ABOVE_BONUS=5.0
export STACK_ABOVE_XY_TOL=0.22
export STACK_ABOVE_Z_TOL=0.14
export STACK_TOP_RELEASE_PROGRESS_W=0.50
export STACK_TOP_HOLD_PEN_W=0.25
export STACK_TOP_HOLD_GRACE_STEPS=5
export STACK_TOP_REQUIRE_HAND_CLEAR=1
export STACK_TOP_HAND_CLEAR=0.12
export STACK_TOP_XY_TOL=0.16
export STACK_TOP_Z_TOL=0.10
export STACK_TOP_STABLE_LIN=0.20
export STACK_TOP_STABLE_ANG=0.50
export STACK_TOP_UPRIGHT_DEG=15
export STACK_TOP_STEPS=5
export STACK_SUCCESS_BONUS=20.0
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_RELEASE_DONE_REWARD=1.00
export STACK_CLEAR_DONE_REWARD=0.00

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
