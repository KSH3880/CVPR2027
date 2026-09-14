#!/bin/bash
# ms18 e9000에서 W2S 정지와 shared-goal late-STACK rehearsal을 한 번에 학습한다.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms52_ms18init_a2trans_w2s_steer100_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms57_ms18e9000_sharedgoal_w2s_boot15_6000_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_MB=${PILOT_MB:-16384}
export MS_ITERS=${PILOT_ITERS:-6000}
export MS_LR=${PILOT_LR:-5e-6}
export MS_SAVE_LATEST=${PILOT_SAVE_EVERY:-100}
export MS_SAVE_ARCHIVE=${PILOT_SAVE_EVERY:-100}
export MS_EPISODE_LENGTH=${PILOT_EPISODE_LENGTH:-780}
export MA_GPU=${MA_GPU:-7}

# ms55에서 carry가 무너진 shared residual 변화까지 차단한다. Actor는 기존
# steering tokenizer만 열고 carry/teammate/backbone/residual/RMS를 고정한다.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FINETUNE_NEWCARRY_RESIDUAL=0
export MA_FINETUNE_STEER_ONLY=1
export MA_FREEZE_INPUT_RMS=1
export MA_TOKEN=mask
export MA_C=0
export MA_BETA=0

# 두 역할의 task-level XY goal은 같다. Top은 시작 즉시 box를 들고 goal 앞
# 0.9 m safety gate까지 운반한 뒤, Base가 15-frame 정지를 마칠 때까지 HOLD한다.
export STACK_SHARED_GOAL_CARRY=1
export STACK_SHARED_WAIT_DIST=0.90
export STACK_TOP_WAIT_AT_START=0
export STACK_STAGE_Z=0.90
export STACK_STAGE_USE_HAND_Z=0
export STACK_STAGE_FORCE_ZERO=1
export STACK_STAGE_TOL=0.35
export STACK_STAGE_HAND_TOL=0.25
export STACK_STAGE_STABLE_LIN=0.25
export STACK_STAGE_STABLE_ANG=1.0
export STACK_STAGE_HOLD_STEPS=5
export STACK_REQUIRE_STAGED=1
export STACK_TOP_COMMIT_GOAL=1
export STACK_TOP_CARRY_TARGET_ONLY=0
unset STACK_DEBUG_TOP_CARRY_TARGET_ONLY
export STACK_TOP_DIRECT_CARRY_REWARD=1
export STACK_TOP_SCALE=1.0

# Base는 release endpoint에서 감속·정지하고 STACK 동안 그 위치를 유지한다.
# 두 dense HOLD 보상은 frame당 최대 0.1이라 40점 strict bonus를 압도하지 않는다.
export STACK_CLEAR_STOP_ON_STACK=1
export STACK_STOP_DECEL_DIST=0.30
export STACK_STOP_HOLD_STEPS=15
export STACK_STOP_LIN=0.10
export STACK_STOP_ANG=0.50
export STACK_STOP_UPRIGHT_DEG=15.0
export STACK_STOP_CONTACT_FORCE=1.0
export STACK_STOP_REWARD_W=1.0
export STACK_TOP_WAIT_REWARD_W=0.10
export STACK_BASE_HOLD_REWARD_W=0.10

# 한 run 안에서 full sequence가 주류이고, carry 보존과 late STACK 노출을
# 각각 rehearsal/snapshot reset으로 섞는다. 전체 reset의 20%가 carry이고,
# 나머지 80% 중 18.75%를 bootstrap으로 뽑아 전체 기대 비율 15%를 맞춘다.
# Snapshot은 현재 rollout의 물리 state만 저장/복원하며 과거 (s,a,r)를 replay하지 않는다.
# 기존 stage gate가 grasp+box 안정 5 frame을 이미 요구하므로 snapshot bank를
# 비우게 만들었던 중복 Top balance streak는 사용하지 않는다.
export STACK_REHEARSAL_FRAC=0.20
export STACK_BOOTSTRAP_FRAC=${PILOT_BOOTSTRAP_FRAC:-0.1875}
export STACK_BOOTSTRAP_KEEP_WAIT=0
export STACK_BOOTSTRAP_CAPTURE_STABLE=1
export STACK_BOOTSTRAP_TOP_BALANCE_STEPS=0
export STACK_BOOTSTRAP_TOP_ROOT_LIN=0.10
export STACK_BOOTSTRAP_TOP_ROOT_ANG=0.50
export STACK_BOOTSTRAP_TOP_UPRIGHT_DEG=15.0
export STACK_BOOTSTRAP_TOP_BOX_LIN=0.15
export STACK_BOOTSTRAP_TOP_BOX_ANG=0.40
export STACK_BOOTSTRAP_TOP_REQUIRE_GRASP=1
export STACK_BOOTSTRAP_EVAL=0

# Base CLEAR에서는 carry 입력을 attention에서 제외해 열린 steering tokenizer가
# 후퇴/감속/HOLD에 집중하게 한다. Top의 carry token과 target은 계속 live다.
export STACK_VIRTUAL_RETREAT_BOX=0
export STACK_VIRTUAL_RETREAT_REAR_BOX=0
export STACK_ZERO_CARRY_OBS=1
export STACK_DYNAMIC_CARRY_MASK=1
export STACK_SEQUENTIAL_REWARD_MASK=1

# 큰 중간 bonus 없이 dense shaping + strict final success 40만 사용한다.
export STACK_TRANSITION_BONUS=0.0
export STACK_CLEAR_BONUS=0.0
export STACK_ABOVE_BONUS=0.0
export STACK_SUCCESS_BONUS=40.0

unset MS_METRICS MA_METRICS
export MS_GRADCHK=1

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
