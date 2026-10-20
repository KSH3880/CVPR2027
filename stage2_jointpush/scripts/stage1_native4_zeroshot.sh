#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-eval}
case "$MODE" in eval|view) ;; *) echo 'Usage: bash stage2_jointpush/scripts/stage1_native4_zeroshot.sh [eval|view]' >&2; exit 2;; esac
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
source "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh"
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
PACKAGE="$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/native4_zeroshot"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/stage1_native4_${MODE}_$(date +%Y%m%d_%H%M%S)}
case "$OUT" in /*) ;; *) OUT="$PROJECT_ROOT/$OUT";; esac
mkdir -p "$OUT"
SOURCE=$(python "$PACKAGE/setup.py" "$OUT" "${STAGE1_CHECKPOINT:-}")
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
export PYTHONUNBUFFERED=1 PROBE_AGENTS=4 PROBE_SPAWN=native PROBE_OUT="$OUT/result.json"
export PROBE_STEPS=${PROBE_STEPS:-300}
set -- --task HumanoidMAOpenPush --cfg_env "$OUT/env.yaml" --cfg_train "$OUT/train.yaml" \
  --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
  --num_agents 4 --num_objects 8 --output_path "$OUT/evaluation" --experiment Stage1Native4 \
  --no_video --task_graph push --test --eval --eval_skills loco --eval_skill_probs 1.0 \
  --checkpoint "$SOURCE" --seed "${SEED:-123}"
if [ "$MODE" = view ]; then
  export PROBE_VIEW=1
  if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -z "${VK_DRIVER_FILES:-}" ] && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  fi
  set -- "$@" --num_envs "${NUM_ENVS:-1}" --eval_repeats "${REPEATS:-1000000}"
else
  export PROBE_VIEW=0
  set -- "$@" --num_envs "${NUM_ENVS:-64}" --eval_repeats 1 --headless
fi
printf 'Stage1 native4 %s output: %s\n' "$MODE" "$OUT"
exec python "$PACKAGE/launch.py" "$@"
