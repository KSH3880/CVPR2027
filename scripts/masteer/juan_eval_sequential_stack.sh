#!/bin/bash
# Juan scoring contract on the local HumanoidMASequentialStackRelease scenario.
# Usage: juan_eval_sequential_stack.sh <source-tag> <policy.pth> [eval-tag]
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE=${1:?usage: juan_eval_sequential_stack.sh <source-tag> <policy.pth> [eval-tag]}
POLICY=${2:?usage: juan_eval_sequential_stack.sh <source-tag> <policy.pth> [eval-tag]}
TAG=${3:-$(basename "$POLICY" .pth)}
GPU=${MA_GPU:-7}
ENVS=${STACK_EVAL_ENVS:-270}
BOX_GRID=${STACK_EVAL_BOX_GRID:-1}
ENV_FILE="$ROOT/runs/queue/logs/$SOURCE.env"
TRAIN_ENV="$ROOT/runs/gen_cfgs/masteer/$SOURCE.yaml"
RESULT_DIR="$ROOT/runs/results/masteer"
METRICS="$RESULT_DIR/juan_eval_${TAG}.npy"
LOG="$RESULT_DIR/juan_eval_${TAG}.log"
EVAL_ENV="$ROOT/runs/gen_cfgs/masteer/juan_eval_${TAG}.yaml"
CLAIM="$ROOT/runs/queue/gpu_locks/juan_eval_$TAG"

case "$GPU" in 6|7) ;; *) echo "MA_GPU must be 6 or 7" >&2; exit 2 ;; esac
case "$TAG" in
    ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2 ;;
esac
case "$ENVS" in
    ""|*[!0-9]*) echo "STACK_EVAL_ENVS must be a positive integer" >&2; exit 2 ;;
esac
[ "$ENVS" -gt 0 ] || exit 2
if [ "$BOX_GRID" != 0 ] && [ $((ENVS % 9)) -ne 0 ]; then
    echo "3x3 box grid requires STACK_EVAL_ENVS divisible by 9" >&2
    exit 2
fi
for file in "$POLICY" "$ENV_FILE" "$TRAIN_ENV"; do
    [ -f "$file" ] || { echo "required file not found: $file" >&2; exit 4; }
done
[ ! -e "$METRICS" ] && [ ! -e "$LOG" ] || {
    echo "result exists; use another eval tag: $TAG" >&2
    exit 3
}

mkdir -p "$RESULT_DIR" "$(dirname "$EVAL_ENV")" "$(dirname "$CLAIM")"
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "juan eval is already claimed: $TAG" >&2
    exit 3
fi
printf '%s\n' "$GPU" > "$CLAIM/gpu"
cleanup() {
    rm -f -- "$CLAIM/gpu"
    rmdir "$CLAIM" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source "$ENV_FILE"
while IFS= read -r name; do
    if [ -z "${!name}" ]; then
        unset "$name"
    fi
done < <(sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$ENV_FILE")

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export MA_METRICS="$METRICS"
export MA_EVAL_ALL_AGENT_ROWS=1
export MA_EVAL_FLUSH_DONE=1
export EVAL_NUM_REPEAT=3
export STACK_EVAL_BOX_GRID="$BOX_GRID"
export STACK_EVAL_BOX_SIZE_IDS=${STACK_EVAL_BOX_SIZE_IDS:-0,4,7}
export STACK_EVAL_SUCCESS_MODE=${STACK_EVAL_SUCCESS_MODE:-box_radius}
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-900}
export STACK_BOTTOM_Z_TOL=${STACK_BOTTOM_Z_TOL:-0.05}
export STACK_BOTTOM_DISPLACE_TOL=${STACK_BOTTOM_DISPLACE_TOL:-0.50}
export STACK_TOP_Z_TOL=${STACK_TOP_Z_TOL:-0.08}
export STACK_DROP_XY="$STACK_BOTTOM_DISPLACE_TOL"
export STACK_REHEARSAL_FRAC=0
export STACK_BOOTSTRAP_FRAC=0

python3 - "$TRAIN_ENV" "$EVAL_ENV" "$ENVS" "$STACK_EPISODE_LENGTH" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
text = re.sub(r"^  numAgents:.*$", "  numAgents: 2", text, flags=re.M)
text = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[3]}", text, flags=re.M)
text = re.sub(
    r"^  episodeLength:.*$", f"  episodeLength: {sys.argv[4]}",
    text, flags=re.M)
text = re.sub(
    r"^  episodeLengthShort:.*$", f"  episodeLengthShort: {sys.argv[4]}",
    text, flags=re.M)
open(sys.argv[2], "w").write(text)
PY

TRAIN_CFG=${MS_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml}
if [ -f "$ROOT/runs/gen_cfgs/masteer/${SOURCE}_train.yaml" ]; then
    TRAIN_CFG="$ROOT/runs/gen_cfgs/masteer/${SOURCE}_train.yaml"
fi
STAGE1=${MS_CKPT:-$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth}
[ -f "$STAGE1" ] || { echo "stage-1 checkpoint not found: $STAGE1" >&2; exit 4; }

if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE=$("$CONDA_EXE" info --base)
    else
        CONDA_BASE=$(conda info --base)
    fi
fi
set +u
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi_koo}"
set -u

echo "juan local-env eval: source=$SOURCE policy=$POLICY envs=$ENVS gpu=$GPU" | tee "$LOG"
cd "$ROOT/TokenHSI-masteer"
set +e
python -u ./tokenhsi/run.py \
    --task HumanoidMAJuanEvalSequentialStackRelease \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 \
    --physx --pipeline gpu --headless \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$EVAL_ENV" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs "$ENVS" --seed "${STACK_EVAL_SEED:-0}" \
    --test --eval --eval_task carry 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
    echo "JUAN_EVAL_ERROR tag=$TAG rc=$rc" | tee -a "$LOG"
    exit "$rc"
fi
[ -f "$METRICS" ] || { echo "metric file was not created: $METRICS" >&2; exit 5; }

cd "$ROOT"
python3 scripts/masteer/juan_eval_sequential_stack_summary.py \
    "$TAG" "$METRICS" "$ENVS" "$BOX_GRID" | tee -a "$LOG"
