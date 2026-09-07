#!/bin/bash
# Deterministic closed-loop evaluation of one coordinator PTH with frozen ms18.
#
#   bash scripts/coord/eval_one.sh c1_cross_s0 cross 128 0
#   COORD_SOURCE=c1_cross_s0 bash scripts/coord/eval_one.sh \
#     runs/coord/c1_cross_s0/coord_c1_000100.pth free 128 0
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-coord"
INPUT=${1:?사용법: eval_one.sh <coord tag 또는 PTH> <free|cross> [envs] [seed]}
SCENARIO=${2:?사용법: eval_one.sh <coord tag 또는 PTH> <free|cross> [envs] [seed]}
ENVS=${3:-128}
SEED=${4:-0}
PROVIDER=${COORD_EVAL_PROVIDER:-learned}

case "$SCENARIO" in free|cross) ;; *) echo "scenario는 free 또는 cross: $SCENARIO" >&2; exit 2 ;; esac
case "$PROVIDER" in learned|analytic) ;; *) echo "COORD_EVAL_PROVIDER는 learned 또는 analytic" >&2; exit 2 ;; esac
if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]]; then echo "envs는 양의 정수: $ENVS" >&2; exit 2; fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then echo "seed는 음이 아닌 정수: $SEED" >&2; exit 2; fi

resolve_file() {
    local value=$1 candidate
    for candidate in "$value" "$ROOT/$value" "$REPO/$value"; do
        if [ -f "$candidate" ]; then realpath "$candidate"; return 0; fi
    done
    return 1
}

if COORD_CKPT=$(resolve_file "$INPUT"); then
    SOURCE=${COORD_SOURCE:?PTH 직접 지정 시 COORD_SOURCE=<학습 tag>가 필요하다}
    CKPT_NAME=$(basename "$COORD_CKPT" .pth)
else
    SOURCE=$INPUT
    if [ -f "$ROOT/runs/coord/$SOURCE/coord_c2_latest.pth" ]; then
        COORD_CKPT="$ROOT/runs/coord/$SOURCE/coord_c2_latest.pth"
    elif [ -f "$ROOT/runs/coord/$SOURCE/coord_b0_latest.pth" ]; then
        COORD_CKPT="$ROOT/runs/coord/$SOURCE/coord_b0_latest.pth"
    else
        COORD_CKPT="$ROOT/runs/coord/$SOURCE/coord_c1_latest.pth"
    fi
    CKPT_NAME=latest
fi
if [ ! -f "$COORD_CKPT" ]; then echo "coordinator PTH 없음: $COORD_CKPT" >&2; exit 3; fi

ENV_FILE="$ROOT/runs/queue/logs/$SOURCE.env"
TRAIN_ENV="$ROOT/runs/gen_cfgs/coord/$SOURCE.yaml"
if [ ! -f "$ENV_FILE" ] || [ ! -f "$TRAIN_ENV" ]; then
    echo "학습 sidecar/cfg 없음: $SOURCE" >&2
    exit 3
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${COORD_EXEC_CKPT:?sidecar에 frozen ms18 PTH가 없다}"
: "${MS_CKPT:?sidecar에 stage1 PTH가 없다}"
if [ ! -f "$COORD_EXEC_CKPT" ] || [ ! -f "$MS_CKPT" ]; then
    echo "executor 또는 stage1 PTH가 없다" >&2; exit 3
fi

SAFE_SOURCE=$(printf '%s' "$SOURCE" | tr -c 'A-Za-z0-9_.-' '_')
EVAL_ID="${SAFE_SOURCE}_${CKPT_NAME}_${PROVIDER}_${SCENARIO}_s${SEED}"
OUT_DIR="$ROOT/runs/results/coord/$SAFE_SOURCE"
LOG="$OUT_DIR/${EVAL_ID}.log"
METRICS="$OUT_DIR/${EVAL_ID}.npy"
SUMMARY="$OUT_DIR/${EVAL_ID}.json"
LOCK="$ROOT/runs/queue/gpu_locks/coord_eval_${EVAL_ID}"
mkdir -p "$OUT_DIR" "$ROOT/runs/queue/gpu_locks" "$ROOT/runs/gen_cfgs/coord"
if ! mkdir "$LOCK" 2>/dev/null; then echo "이미 평가 중: $EVAL_ID" >&2; exit 4; fi

TMP_DIR=$(mktemp -d "/tmp/coord_eval_${SAFE_SOURCE}.XXXXXX")
EVAL_ENV=$(mktemp "/tmp/coord_eval_${SAFE_SOURCE}.XXXXXX.yaml")
cleanup() {
    rm -rf -- "$LOCK" "$TMP_DIR"
    rm -f -- "$EVAL_ENV"
}
trap cleanup EXIT INT TERM
mkdir -p "$TMP_DIR/nn"
EXEC_SNAPSHOT="$TMP_DIR/nn/Humanoid.pth"
cp -- "$COORD_EXEC_CKPT" "$EXEC_SNAPSHOT"
sed -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" "$TRAIN_ENV" > "$EVAL_ENV"
rm -f -- "$METRICS" "$SUMMARY"

if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    elif [ -f /home/cvlab/anaconda3/etc/profile.d/conda.sh ]; then CONDA_BASE=/home/cvlab/anaconda3
    else echo "conda를 찾지 못했다" >&2; exit 1
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"

export COORD_PROVIDER=$PROVIDER COORD_CKPT COORD_DRAW_CANDIDATES=0
export COORD_REPLAN_STEPS=${COORD_REPLAN_STEPS:-6}
export COORD_SPEED_LIMIT=${COORD_SPEED_LIMIT:-1}
export MS_SCEN=$SCENARIO MS_SEED=$SEED MS_DBG=0
export MS_METRICS=$METRICS MA_METRICS=$METRICS
unset MA_LAYOUT MA_LAYOUT_D MA_LAYOUT_S MA_LAYOUT_L MS_SCEN_CURVE MS_VIZ

cd "$REPO"
set +e
python -u ./tokenhsi/run.py \
    --task HumanoidMACoordCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$EVAL_ENV" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$MS_CKPT" --checkpoint "$EXEC_SNAPSHOT" \
    --num_envs "$ENVS" --headless --seed "$SEED" \
    --test --eval --eval_task carry > "$LOG" 2>&1
run_code=$?
set -e
if [ "$run_code" -ne 0 ] || [ ! -f "$METRICS" ]; then
    echo "COORD_EVAL_ERROR id=$EVAL_ID rc=$run_code metrics=$([ -f "$METRICS" ] && echo yes || echo no)" | tee -a "$LOG" >&2
    if [ "$run_code" -ne 0 ]; then exit "$run_code"; fi
    exit 5
fi

python "$ROOT/scripts/coord/summarize_eval.py" \
    --tag "$SOURCE/$CKPT_NAME/$PROVIDER" --scenario "$SCENARIO" \
    --checkpoint "$COORD_CKPT" --log "$LOG" --metrics "$METRICS" --output "$SUMMARY" \
    | tee -a "$LOG"
