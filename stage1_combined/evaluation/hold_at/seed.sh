#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
RUN_DIR=$(cd "$ROOT/.." && pwd)
SEED=$1
NUM_ENVS=$2
export TOKENHSI_GPU=${TOKENHSI_GPU:-5} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh} PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTORCH_JIT=${PYTORCH_JIT:-0}
unset HOLD_AT_OBJECT_ROOT CHECK MAX_ITERATIONS RESUME_CHECKPOINT
. "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export OUTPUT_PATH="$ROOT/output_etc/check/hold_at_seed$SEED"
mkdir -p "$OUTPUT_PATH"
cp "$RUN_DIR/env.yaml" "$OUTPUT_PATH/resolved_env.yaml"
MOTION_FILE=tokenhsi/data/dataset_tokenhsi_hold_at.yaml
if [ ! -f "$MOTION_FILE" ]; then MOTION_FILE=tokenhsi/data/dataset_table_hold_at.yaml; fi
"${TOKENHSI_EVAL_PYTHON:-python}" -u evaluation/hold_at/worker.py --task HumanoidChairHoldAtFT \
 --cfg_env "$OUTPUT_PATH/resolved_env.yaml" --cfg_train "$RUN_DIR/train.yaml" \
 --checkpoint "$RUN_DIR/checkpoint.pth" --motion_file "$MOTION_FILE" \
 --num_agents 1 --num_objects 1 --num_envs "$NUM_ENVS" \
 --test --eval --headless --no_video --eval_repeats 1 \
 --eval_skills loco --eval_skill_probs 1.0 --task_graph holding_at \
 --seed "$SEED" --episode_length 900 --output_path "$OUTPUT_PATH"
