#!/bin/bash
# Retrain the proven ms52 W2S contract in the shared-goal scenario.
# Carry tokenizer stays frozen; steering extras and the ms52 residual stay trainable.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
BASE_TAG=ms52_ms18init_a2trans_w2s_steer100_3000_s0
BASE_ENV=${PILOT_BASE_ENV:-$ROOT/runs/queue/logs/$BASE_TAG.env}
if [ ! -f "$BASE_ENV" ]; then
    LOCAL_FALLBACK="$ROOT/runs/queue/logs/ms38_ms18init_seqrewardmask_side135_dynmask_s0.env"
    if [ -f "$LOCAL_FALLBACK" ]; then
        echo "ms52 sidecar 없음; 로컬 ms38 sidecar + 명시적 ms58 override 사용: $LOCAL_FALLBACK" >&2
        BASE_ENV="$LOCAL_FALLBACK"
    fi
fi
INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
TAG=${1:-ms58_ms18e9000_ms52_sharedgoal_safe_w2s_3000_s0}

if [ ! -f "$BASE_ENV" ] || [ ! -f "$INIT_CKPT" ]; then
    echo "기준 파일 없음: $BASE_ENV 또는 $INIT_CKPT" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_INIT_CKPT="$INIT_CKPT"
export MS_CKPT="$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth"
export MS_TRAINCFG=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_MB=${PILOT_MB:-16384}
export MS_ITERS=${PILOT_ITERS:-3000}
export MS_SAVE_LATEST=${PILOT_SAVE_EVERY:-100}
export MS_SAVE_ARCHIVE=${PILOT_SAVE_EVERY:-100}
export MS_EPISODE_LENGTH=${PILOT_EPISODE_LENGTH:-780}
export MA_GPU=${PILOT_GPU:-${MA_GPU:-7}}

# Preserve the exact ms52 actor contract: carry tokenizer frozen, steering
# extras and the existing internal action residual trainable.
export MA_NOFREEZE=0
export MA_ADAPTER_ONLY=0
export MA_FREEZE_NEW_CARRY=1
export MA_FREEZE_INPUT_RMS=1
unset MA_FINETUNE_NEWCARRY_RESIDUAL MA_FINETUNE_STEER_ONLY

# Adapt only the task geometry. Both boxes share the task-level XY goal; Top
# carries to a safety gate selected from paired path candidates and waits there.
export STACK_SHARED_GOAL_CARRY=1
export STACK_SHARED_WAIT_DIST=${PILOT_SHARED_WAIT_DIST:-1.0}
export STACK_SHARED_PATH_CLEARANCE=${PILOT_PATH_CLEARANCE:-0.8}
export STACK_SHARED_PATH_CANDIDATES=${PILOT_PATH_CANDIDATES:-8}
export STACK_SHARED_PATH_RETRIES=${PILOT_PATH_RETRIES:-4}
export STACK_SHARED_PATH_START_MARGIN=${PILOT_PATH_START_MARGIN:-0.6}
export STACK_TOP_WAIT_REWARD_W=${PILOT_TOP_WAIT_REWARD_W:-${STACK_TOP_WAIT_REWARD_W:-0.0}}
export STACK_BASE_HOLD_REWARD_W=${PILOT_BASE_HOLD_REWARD_W:-${STACK_BASE_HOLD_REWARD_W:-0.0}}
export STACK_STAGE_USE_HAND_Z=0
export STACK_STAGE_FORCE_ZERO=1
export STACK_STAGE_TOL=0.35
export STACK_STAGE_HAND_TOL=0.25
export STACK_STAGE_STABLE_LIN=0.25
export STACK_STAGE_STABLE_ANG=1.0
export STACK_STAGE_HOLD_STEPS=5
export STACK_REQUIRE_STAGED=1
export STACK_TOP_WAIT_AT_START=0
export STACK_TOP_COMMIT_GOAL=1
export STACK_TOP_CARRY_TARGET_ONLY=0
export STACK_TOP_DIRECT_CARRY_REWARD=0
export STACK_TOP_SCALE=1.0

# Preserve ms52 W2S shaping and full-sequence learning. Only the consecutive
# stop streak is relaxed from 15 frames (0.5 s) to 10 frames (about 0.33 s).
export STACK_CLEAR_STEER_W=1.0
export STACK_CLEAR_STOP_ON_STACK=1
export STACK_STOP_DECEL_DIST=${PILOT_STOP_DECEL_DIST:-0.30}
export STACK_STOP_HOLD_STEPS=${PILOT_STOP_HOLD_STEPS:-10}
export STACK_STOP_LIN=${PILOT_STOP_LIN:-0.10}
export STACK_STOP_ANG=${PILOT_STOP_ANG:-0.50}
export STACK_STOP_UPRIGHT_DEG=${PILOT_STOP_UPRIGHT_DEG:-15.0}
export STACK_STOP_CONTACT_FORCE=${PILOT_STOP_CONTACT_FORCE:-1.0}
export STACK_STOP_REWARD_W=1.0
export STACK_BOOTSTRAP_FRAC=0.0
export STACK_BOOTSTRAP_KEEP_WAIT=0
export STACK_REHEARSAL_FRAC=0.10

# Make the inherited signed CLEAR contract explicit so the local fallback and
# the original ms52 sidecar produce the same retreat learning signal.
export STACK_NEGATIVE_CLEAR_REWARD=${PILOT_NEGATIVE_CLEAR_REWARD:-1}
export STACK_CLEAR_CLASSIC_STEER=${PILOT_CLEAR_CLASSIC_STEER:-0}
export STACK_CLEAR_CLASSIC_STOP_REWARD=${PILOT_CLEAR_CLASSIC_STOP_REWARD:-0}
export STACK_CLEAR_HEADING_PROGRESS_W=${PILOT_CLEAR_HEADING_PROGRESS_W:-0.0}
export STACK_CLEAR_MOVE_W=${PILOT_CLEAR_MOVE_W:-1.0}
export STACK_CLEAR_HAND_PEN_W=0.5
export STACK_CLEAR_STALL_PEN_W=${PILOT_CLEAR_STALL_PEN_W:-1.5}
export STACK_CLEAR_PATH_PEN_W=${PILOT_CLEAR_PATH_PEN_W:-0.0}
export STACK_CLEAR_REVERSE_PEN_W=0.5
export STACK_CLEAR_FORWARD_GATE=${PILOT_CLEAR_FORWARD_GATE:-0}
export STACK_BASE_REGRASP_PEN_W=${PILOT_BASE_REGRASP_PEN_W:-0.0}
export STACK_CLEAR_MOVE_MIN_FRAC=0.40
export STACK_CLEAR_GRACE_STEPS=5
export STACK_SEQUENTIAL_REWARD_MASK=1
export STACK_CARRY_DONE_REWARD=1.0
export STACK_RELEASE_DONE_REWARD=0.5
export STACK_CLEAR_DONE_REWARD=0.0

# Optional CLEAR controls are supplied by focused experiment wrappers. Keep
# ms58's historical defaults when they are absent.
export STACK_RETREAT_SIDE_DEG=${PILOT_RETREAT_SIDE_DEG:-${STACK_RETREAT_SIDE_DEG:-0}}
export STACK_RETREAT_SIDE_BINS=${PILOT_RETREAT_SIDE_BINS:-${STACK_RETREAT_SIDE_BINS:-1}}
export STACK_RETREAT_RANDOM=${PILOT_RETREAT_RANDOM:-${STACK_RETREAT_RANDOM:-0}}
export STACK_RETREAT_DIST=${PILOT_RETREAT_DIST:-${STACK_RETREAT_DIST:-1.5}}
export STACK_RETREAT_DIST_MIN=${PILOT_RETREAT_DIST_MIN:-${STACK_RETREAT_DIST_MIN:-$STACK_RETREAT_DIST}}
export STACK_VIRTUAL_RETREAT_BOX=${PILOT_VIRTUAL_RETREAT_BOX:-${STACK_VIRTUAL_RETREAT_BOX:-0}}
export STACK_VIRTUAL_RETREAT_REAR_BOX=${PILOT_VIRTUAL_RETREAT_REAR_BOX:-${STACK_VIRTUAL_RETREAT_REAR_BOX:-0}}
export STACK_ZERO_CARRY_OBS=${PILOT_ZERO_CARRY_OBS:-${STACK_ZERO_CARRY_OBS:-0}}
export STACK_DYNAMIC_CARRY_MASK=${PILOT_DYNAMIC_CARRY_MASK:-${STACK_DYNAMIC_CARRY_MASK:-0}}
export STACK_CARRY_OBS_ZERO_FADE=${PILOT_CARRY_OBS_ZERO_FADE:-${STACK_CARRY_OBS_ZERO_FADE:-0}}
export STACK_CARRY_OBS_ZERO_FADE_STEPS=${PILOT_CARRY_OBS_ZERO_FADE_STEPS:-0}
export STACK_CLEAR_ARC_DIST=${PILOT_CLEAR_ARC_DIST:-${STACK_CLEAR_ARC_DIST:-0.60}}

# The terminal +20 is paid to each agent only after stable placement and
# base/top face alignment. Cube quarter-turns pass; a 45-degree diamond fails.
export STACK_SUCCESS_BONUS=${PILOT_SUCCESS_BONUS:-20.0}
export STACK_TOP_PARALLEL_DEG=${PILOT_TOP_PARALLEL_DEG:-15.0}

export MS_GRADCHK=1
unset MS_METRICS MA_METRICS

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
