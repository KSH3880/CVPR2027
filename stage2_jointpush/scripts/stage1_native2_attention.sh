#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
source "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh"
PACKAGE="$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/native4_zeroshot"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/native2_attention_$(date +%Y%m%d_%H%M%S)}
case "$OUT" in /*) ;; *) OUT="$PROJECT_ROOT/$OUT";; esac
mkdir -p "$OUT"
SOURCE=$(python "$PACKAGE/setup.py" "$OUT" "${STAGE1_CHECKPOINT:-}" --agents 2)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
export NATIVE2_ATTENTION_CAPTURE=1 ATTENTION_OUT="$OUT" PROBE_STEPS=${PROBE_STEPS:-300}
export PYTHONUNBUFFERED=1
exec python "$PACKAGE/native2_launch.py" --task HumanoidMAOpenPush \
 --cfg_env "$OUT/env.yaml" --cfg_train "$OUT/train.yaml" \
 --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
 --num_agents 2 --num_objects 4 --num_envs "${NUM_ENVS:-4}" \
 --output_path "$OUT/evaluation" --experiment Stage1Native2Attention \
 --no_video --task_graph "${TASK_GRAPH:-random_open_push}" --test --eval --headless --eval_repeats 1 \
 --eval_skills loco --eval_skill_probs 1.0 --checkpoint "$SOURCE" --seed "${SEED:-123}"
