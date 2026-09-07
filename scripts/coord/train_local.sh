#!/bin/bash
# High-level coordinator PPO. The ms18 executor is loaded for inference and never enters
# the optimizer; only TokenHSI-coord/coordinator parameters are updated.
#
# COORD_ITERS=200 COORD_ENVS=64 \
#   bash scripts/coord/train_local.sh c1_s0 ms18_maskteam_origscale_c06_s0
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-coord"
EXEC_REPO=${COORD_EXEC_REPO:-$ROOT/TokenHSI-masteer}
TAG=${1:?사용법: train_local.sh <tag> <ms18 tag 또는 PTH>}
EXEC_INPUT=${2:?사용법: train_local.sh <tag> <ms18 tag 또는 PTH>}
ENVS=${COORD_ENVS:-64}
SEED=${COORD_SEED:-0}

if ! [[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "잘못된 tag: $TAG" >&2
    exit 2
fi
if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "COORD_ENVS는 양의 정수여야 한다: $ENVS" >&2
    exit 2
fi

resolve_file() {
    local value=$1 candidate
    for candidate in "$value" "$ROOT/$value" "$REPO/$value" "$EXEC_REPO/$value"; do
        if [ -f "$candidate" ]; then realpath "$candidate"; return 0; fi
    done
    return 1
}

if ! EXEC_CKPT=$(resolve_file "$EXEC_INPUT"); then
    EXEC_CKPT=$(find "$EXEC_REPO/output/masteer/$EXEC_INPUT" -type f -name Humanoid.pth \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [ -z "$EXEC_CKPT" ] || [ ! -f "$EXEC_CKPT" ]; then
    echo "ms18 체크포인트 없음: $EXEC_INPUT" >&2
    exit 1
fi

if [ -n "${MS_CKPT:-}" ]; then
    BASE_CKPT=$(resolve_file "$MS_CKPT") || {
        echo "stage1 체크포인트 없음: $MS_CKPT" >&2; exit 1;
    }
else
    BASE_CKPT=""
    for candidate in "$ROOT/../TokenHSI/output/tokenhsi/ckpt_stage1.pth" \
                     "$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth"; do
        if [ -f "$candidate" ]; then BASE_CKPT=$(realpath "$candidate"); break; fi
    done
fi
if [ -z "$BASE_CKPT" ]; then
    echo "stage1 체크포인트 없음. MS_CKPT=<ckpt_stage1.pth>를 지정한다." >&2
    exit 1
fi

if [ -n "${COORD_INIT:-}" ]; then
    COORD_INIT_INPUT=$COORD_INIT
    if ! COORD_INIT=$(resolve_file "$COORD_INIT_INPUT"); then
        echo "초기 coordinator 체크포인트 없음: $COORD_INIT_INPUT" >&2
        exit 1
    fi
    export COORD_INIT
fi

OUT_DIR="$ROOT/runs/coord/$TAG"
if [ -e "$OUT_DIR" ]; then
    if [ "${COORD_RESUME_IN_PLACE:-0}" != 1 ] || [ -z "${COORD_INIT:-}" ]; then
        echo "같은 tag 출력이 이미 있다: $OUT_DIR" >&2
        exit 3
    fi
else
    mkdir -p "$OUT_DIR"
fi
mkdir -p "$ROOT/runs/gen_cfgs/coord" "$ROOT/runs/queue/logs"

CFG_SRC="$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
CFG="$ROOT/runs/gen_cfgs/coord/$TAG.yaml"
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' "$CFG_SRC" > "$CFG"

if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    elif [ -f /home/cvlab/anaconda3/etc/profile.d/conda.sh ]; then CONDA_BASE=/home/cvlab/anaconda3
    else echo "conda를 찾지 못했다." >&2; exit 1
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"

export COORD_PROVIDER=external
export COORD_MODEL=${COORD_MODEL:-c1}
case "$COORD_MODEL" in c1|c2|simple) ;; *) echo "COORD_MODEL은 c1, c2 또는 simple" >&2; exit 2 ;; esac
export COORD_OUTPUT="$OUT_DIR"
export COORD_ITERS=${COORD_ITERS:-200}
export COORD_HORIZON=${COORD_HORIZON:-32}
export COORD_LOW_STEPS=${COORD_LOW_STEPS:-6}
export COORD_PPO_EPOCHS=${COORD_PPO_EPOCHS:-3}
export COORD_MINIBATCH=${COORD_MINIBATCH:-512}
export COORD_SAVE_EVERY=${COORD_SAVE_EVERY:-10}
export COORD_LR=${COORD_LR:-0.0003}
export COORD_ANALYTIC_PRIOR=${COORD_ANALYTIC_PRIOR:-0}
export COORD_C2_PEAK_COLLISION=${COORD_C2_PEAK_COLLISION:-0}
export COORD_C2_ACTUAL_INITIAL_SPEED=${COORD_C2_ACTUAL_INITIAL_SPEED:-0}
export COORD_C2_IMMEDIATE_SLOWDOWN=${COORD_C2_IMMEDIATE_SLOWDOWN:-0}
export COORD_C2_HUMAN_CLEARANCE=${COORD_C2_HUMAN_CLEARANCE:-1.0}
export COORD_C2_ARRIVAL_GAP=${COORD_C2_ARRIVAL_GAP:-0}
export COORD_C2_ARRIVAL_TIME_MARGIN=${COORD_C2_ARRIVAL_TIME_MARGIN:-1.0}
export COORD_C2_ARRIVAL_GAP_COEF=${COORD_C2_ARRIVAL_GAP_COEF:-2.0}
export COORD_C2_FIXED_YIELD_A1=${COORD_C2_FIXED_YIELD_A1:-0}
export COORD_C2_DIRECT_SPEED_PROFILE=${COORD_C2_DIRECT_SPEED_PROFILE:-0}
export COORD_C2_BIDIRECTIONAL_TARGET_GAP=${COORD_C2_BIDIRECTIONAL_TARGET_GAP:-0}
export COORD_C2_STATE_ORDERED_TARGET_GAP=${COORD_C2_STATE_ORDERED_TARGET_GAP:-0}
export COORD_C2_ARRIVAL_FEATURES=${COORD_C2_ARRIVAL_FEATURES:-0}
export COORD_C2_MLP_BACKBONE=${COORD_C2_MLP_BACKBONE:-0}
export COORD_C2_MLP_CONFLICT_FEATURES=${COORD_C2_MLP_CONFLICT_FEATURES:-0}
export COORD_C2_DIRECT_WAYPOINTS=${COORD_C2_DIRECT_WAYPOINTS:-0}
export COORD_C2_JOINT_POINT_SPEED=${COORD_C2_JOINT_POINT_SPEED:-0}
export COORD_C2_PHYSICAL_SPEED_CAPS=${COORD_C2_PHYSICAL_SPEED_CAPS:-0}
export COORD_C2_SLOWDOWN_WINDOW=${COORD_C2_SLOWDOWN_WINDOW:-0}
export COORD_C2_SLOWDOWN_WIDTH_MAX=${COORD_C2_SLOWDOWN_WIDTH_MAX:-0}
export COORD_C2_CONFLICT_WINDOW=${COORD_C2_CONFLICT_WINDOW:-0}
export COORD_C2_CONFLICT_MIN_AT_CROSSING=${COORD_C2_CONFLICT_MIN_AT_CROSSING:-0}
export COORD_C2_CONFLICT_PLATEAU=${COORD_C2_CONFLICT_PLATEAU:-0}
export COORD_C2_SMOOTH_DEPTH=${COORD_C2_SMOOTH_DEPTH:-0}
export COORD_C2_WAYPOINT_SMOOTHING_PASSES=${COORD_C2_WAYPOINT_SMOOTHING_PASSES:-0}
export COORD_C2_WAYPOINT_DISTANCE_SCALING=${COORD_C2_WAYPOINT_DISTANCE_SCALING:-0}
export COORD_C2_FIXED_YIELD_SPEED=${COORD_C2_FIXED_YIELD_SPEED:-0}
export COORD_C2_PATH_COLLISION_ONLY=${COORD_C2_PATH_COLLISION_ONLY:-0}
export COORD_C2_COLLISION_FOCUS_STEPS=${COORD_C2_COLLISION_FOCUS_STEPS:-0}
export COORD_C2_COLLISION_TIME_UNCERTAINTY=${COORD_C2_COLLISION_TIME_UNCERTAINTY:-0}
export COORD_C2_PROXIMITY_COLLISION=${COORD_C2_PROXIMITY_COLLISION:-0}
export COORD_C2_PROXIMITY_APPROACH_BETA=${COORD_C2_PROXIMITY_APPROACH_BETA:-1}
export COORD_C2_PROXIMITY_MARGIN=${COORD_C2_PROXIMITY_MARGIN:-0}
export COORD_C2_FUTURE_COLLISION_COEF=${COORD_C2_FUTURE_COLLISION_COEF:-50}
export COORD_C2_SPEED_EFFICIENCY_COEF=${COORD_C2_SPEED_EFFICIENCY_COEF:-0}
export COORD_C2_PATH_RESIDUAL_COEF=${COORD_C2_PATH_RESIDUAL_COEF:-0}
export COORD_C2_PATH_SMOOTH_COEF=${COORD_C2_PATH_SMOOTH_COEF:-0}
export COORD_C2_MEASURED_EXECUTOR_TIMING=${COORD_C2_MEASURED_EXECUTOR_TIMING:-0}
export COORD_C2_CONSISTENCY_COEF=${COORD_C2_CONSISTENCY_COEF:-0}
export COORD_C2_RANDOM_PRIORITY=${COORD_C2_RANDOM_PRIORITY:-0}
export COORD_C2_LEARNED_PRIORITY=${COORD_C2_LEARNED_PRIORITY:-0}
if [ "$COORD_C2_RANDOM_PRIORITY" != 0 ] && [ "$COORD_C2_LEARNED_PRIORITY" != 0 ]; then
    echo "COORD_C2_RANDOM_PRIORITY와 COORD_C2_LEARNED_PRIORITY는 동시에 사용할 수 없다" >&2
    exit 2
fi
export COORD_C2_SPEED_SMOOTH_COEF=${COORD_C2_SPEED_SMOOTH_COEF:-0.5}
export COORD_C2_UNNECESSARY_SLOW_COEF=${COORD_C2_UNNECESSARY_SLOW_COEF:-0.1}
export COORD_C2_EXTRA_DELAY_COEF=${COORD_C2_EXTRA_DELAY_COEF:-0}
export COORD_C2_DETOUR_DELAY_COEF=${COORD_C2_DETOUR_DELAY_COEF:-0}
export COORD_C2_DUAL_DELAY_COEF=${COORD_C2_DUAL_DELAY_COEF:-0}
export COORD_C2_EXPLICIT_GAP_COEF=${COORD_C2_EXPLICIT_GAP_COEF:-0}
export COORD_C2_EXPLICIT_ANCHOR_COEF=${COORD_C2_EXPLICIT_ANCHOR_COEF:-0}
export COORD_C2_EXPLICIT_POST_COEF=${COORD_C2_EXPLICIT_POST_COEF:-0}
export COORD_C2_INITIAL_PLAN_WEIGHT=${COORD_C2_INITIAL_PLAN_WEIGHT:-1}
case "$COORD_MODEL" in
    c1)
        export COORD_AUX_COEF=${COORD_AUX_COEF:-0.01}
        export COORD_COLLISION_COEF=${COORD_COLLISION_COEF:-2}
        export COORD_INVALID_COEF=${COORD_INVALID_COEF:-0.2}
        export COORD_UNSAFE_COEF=${COORD_UNSAFE_COEF:-0.05}
        ;;
    c2)
        export COORD_AUX_COEF=${COORD_AUX_COEF:-1}
        export COORD_COLLISION_COEF=${COORD_COLLISION_COEF:-50}
        export COORD_INVALID_COEF=${COORD_INVALID_COEF:-1}
        export COORD_UNSAFE_COEF=${COORD_UNSAFE_COEF:-0}
        ;;
    simple)
        export COORD_AUX_COEF=${COORD_AUX_COEF:-0}
        export COORD_COLLISION_COEF=${COORD_COLLISION_COEF:-50}
        export COORD_INVALID_COEF=${COORD_INVALID_COEF:-0}
        export COORD_UNSAFE_COEF=${COORD_UNSAFE_COEF:-0}
        ;;
esac
export COORD_SPEED_LIMIT=${COORD_SPEED_LIMIT:-1}
export COORD_DRAW_CANDIDATES=0
export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_SEP=0 MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=0 MS_M_LO=0.25 MS_CLIP=1 MS_SCEN=${MS_SCEN:-free}
export MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_DBG=0

{
    printf 'export COORD_EXEC_CKPT=%q\n' "$EXEC_CKPT"
    printf 'export MS_CKPT=%q\n' "$BASE_CKPT"
    for name in COORD_PROVIDER COORD_MODEL COORD_OUTPUT COORD_ITERS COORD_HORIZON COORD_LOW_STEPS \
        COORD_PPO_EPOCHS COORD_MINIBATCH COORD_SAVE_EVERY COORD_LR COORD_AUX_COEF \
        COORD_ANALYTIC_PRIOR COORD_COLLISION_COEF COORD_INVALID_COEF COORD_UNSAFE_COEF \
        COORD_C2_PEAK_COLLISION COORD_C2_ACTUAL_INITIAL_SPEED COORD_C2_IMMEDIATE_SLOWDOWN \
        COORD_C2_HUMAN_CLEARANCE \
        COORD_C2_ARRIVAL_GAP COORD_C2_ARRIVAL_TIME_MARGIN \
        COORD_C2_ARRIVAL_GAP_COEF \
        COORD_C2_FIXED_YIELD_A1 \
        COORD_C2_DIRECT_SPEED_PROFILE COORD_C2_BIDIRECTIONAL_TARGET_GAP \
        COORD_C2_STATE_ORDERED_TARGET_GAP \
        COORD_C2_ARRIVAL_FEATURES COORD_C2_MLP_BACKBONE \
        COORD_C2_MLP_CONFLICT_FEATURES \
        COORD_C2_DIRECT_WAYPOINTS COORD_C2_JOINT_POINT_SPEED \
        COORD_C2_PHYSICAL_SPEED_CAPS \
        COORD_C2_SLOWDOWN_WINDOW COORD_C2_SLOWDOWN_WIDTH_MAX \
        COORD_C2_CONFLICT_WINDOW \
        COORD_C2_CONFLICT_MIN_AT_CROSSING \
        COORD_C2_CONFLICT_PLATEAU \
        COORD_C2_SMOOTH_DEPTH \
        COORD_C2_WAYPOINT_SMOOTHING_PASSES \
        COORD_C2_WAYPOINT_DISTANCE_SCALING \
        COORD_C2_FIXED_YIELD_SPEED COORD_C2_PATH_COLLISION_ONLY \
        COORD_C2_COLLISION_FOCUS_STEPS COORD_C2_SPEED_EFFICIENCY_COEF \
        COORD_C2_COLLISION_TIME_UNCERTAINTY \
        COORD_C2_PROXIMITY_COLLISION COORD_C2_PROXIMITY_APPROACH_BETA \
        COORD_C2_PROXIMITY_MARGIN \
        COORD_C2_FUTURE_COLLISION_COEF \
        COORD_C2_PATH_RESIDUAL_COEF COORD_C2_PATH_SMOOTH_COEF \
        COORD_C2_MEASURED_EXECUTOR_TIMING \
        COORD_C2_CONSISTENCY_COEF \
        COORD_C2_RANDOM_PRIORITY \
        COORD_C2_LEARNED_PRIORITY \
        COORD_C2_SPEED_SMOOTH_COEF COORD_C2_UNNECESSARY_SLOW_COEF \
        COORD_C2_EXTRA_DELAY_COEF \
        COORD_C2_DETOUR_DELAY_COEF \
        COORD_C2_DUAL_DELAY_COEF \
        COORD_C2_EXPLICIT_GAP_COEF COORD_C2_EXPLICIT_ANCHOR_COEF \
        COORD_C2_EXPLICIT_POST_COEF COORD_C2_INITIAL_PLAN_WEIGHT \
        COORD_SPEED_LIMIT MA_TOKEN MA_TOKENIZER_ZERO MA_SEP MA_SPAWN_GAP \
        MS_MRAND MS_M_LO MS_CLIP MS_SCEN MS_REWARD_OUTER MS_POS_C MS_VEL_W; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
    if [ -n "${COORD_INIT:-}" ]; then printf 'export COORD_INIT=%q\n' "$COORD_INIT"; fi
} > "$ROOT/runs/queue/logs/$TAG.env"

echo "=============================================================="
echo " tag          $TAG"
echo " frozen ms18  $EXEC_CKPT"
echo " env          $ENVS x 2명"
echo " PPO          iter=$COORD_ITERS horizon=$COORD_HORIZON low_steps=$COORD_LOW_STEPS"
echo " output       $OUT_DIR"
echo "=============================================================="

cd "$REPO"
python -u -m coordinator.train_closed_loop \
    --test --headless --task HumanoidMACoordCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$BASE_CKPT" \
    --checkpoint "$EXEC_CKPT" \
    --num_envs "$ENVS" --seed "$SEED" --output_path "$OUT_DIR/ms18_unused"
