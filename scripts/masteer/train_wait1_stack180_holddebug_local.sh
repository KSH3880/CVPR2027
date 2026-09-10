#!/bin/bash
# ms48 wait-goal/STACK-budget pilot: retain the ms47 model/controller/observation
# contract, stage the top at a 1 m fixed WAIT goal, then grant 180 fresh STACK
# frames after the live base pose becomes the top goal. Starts from ms18 epoch 9000.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms45_ms18init_relprogress050_hold050_success10_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms48_ms18init_wait1_stack180_holddebug_1000_s0}

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
export MS_EPISODE_LENGTH=780

# A fixed, reachable wait goal is followed by one retarget to the live base
# pose. STACK receives 180 physics frames after at most 600 pre-STACK frames.
export STACK_STAGE_DIST=1.0
export STACK_STAGE_TOL=0.35
export STACK_STAGE_HAND_TOL=0.25
export STACK_STAGE_STABLE_LIN=0.25
export STACK_STAGE_STABLE_ANG=1.0
export STACK_STAGE_HOLD_STEPS=5
export STACK_REQUIRE_STAGED=1
export STACK_PRE_STEPS=600
export STACK_PHASE_STEPS=180

# Keep holding through WAIT/approach. Release starts only after stable placement.
export STACK_TOP_APPROACH_PROGRESS_W=2.0
export STACK_TOP_PREMATURE_RELEASE_PEN_W=0.50
export STACK_TOP_SETTLE_STEPS=8
export STACK_CLEAR_HARD_GATE=1

# Preserve the exact ms46 model and observation layout. The steering tokenizer
# and internal adapter remain trainable; carry/backbone/composer stay frozen.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
export STACK_DYNAMIC_CARRY_MASK=0
export STACK_ZERO_CARRY_OBS=1
export STACK_REHEARSAL_FRAC=0.65

# Preserve ms46 base release shaping and its curved CLEAR command. Small box
# contact remains soft, and the existing 0.35 m failure margin is unchanged.
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
export STACK_ENTRY_FOOT_GATE=0
export STACK_RELEASE_FOOT_GATE=0
export STACK_FOOT_BOX_W=0.05
export STACK_DROP_XY=0.35

# Reward-only CLEAR correction. RELEASE itself keeps carry_done=1.0. From
# CLEAR onward the completion floor is 1.5, exactly cancelled by a full stall
# penalty after five frames. At the ms46 0.375 m/s command, min_frac=0.40
# restores the 0.15 m/s absolute movement threshold used by successful ms43.
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=1.00
export STACK_RELEASE_DONE_REWARD=0.50
export STACK_CLEAR_DONE_REWARD=0.00
export STACK_NEGATIVE_CLEAR_REWARD=1
export STACK_CLEAR_MOVE_W=1.00
export STACK_CLEAR_HAND_PEN_W=0.50
export STACK_CLEAR_STALL_PEN_W=1.50
export STACK_CLEAR_REVERSE_PEN_W=0.50
export STACK_CLEAR_MOVE_MIN_FRAC=0.40
export STACK_CLEAR_GRACE_STEPS=5

# Keep the ms46 top approach/release shaping. The large one-shot bonus is paid
# only after the full physical success gate is held for five frames, never for
# merely opening the hands above the base box.
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
export STACK_SUCCESS_BONUS=40.0

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
