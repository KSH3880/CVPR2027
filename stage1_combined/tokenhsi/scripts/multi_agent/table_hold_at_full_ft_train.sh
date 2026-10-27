#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-1}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OBJECT_ROOT=${HOLD_AT_OBJECT_ROOT:-$ROOT/tokenhsi/data/dataset_dynamic_objects}
if [ ! -d "$OBJECT_ROOT/train/Table_Circle" ]; then
    echo "Missing dynamic table dataset: $OBJECT_ROOT/train/Table_Circle; set HOLD_AT_OBJECT_ROOT." >&2
    exit 2
fi
export HOLD_AT_OBJECT_ROOT="$OBJECT_ROOT"
. "$SCRIPT_DIR/runtime_env.sh"
. "$SCRIPT_DIR/output_paths.sh"
if [ "${HEADLESS:-1}" = 0 ]; then . "$SCRIPT_DIR/gui_gpu_env.sh"; fi
if [ -z "${OUTPUT_PATH:-}" ]; then
    if [ "${CHECK:-0}" = 1 ]; then
        export OUTPUT_PATH="$ROOT/../runs/table_hold_at_check_$(date +%Y%m%d_%H%M%S)/output_etc/check/train"
    else
        export OUTPUT_PATH="$ROOT/../runs/table_hold_at_$(date +%Y%m%d_%H%M%S)/train"
    fi
fi
tokenhsi_select_output table_hold_at_full_ft train
mkdir -p "$OUTPUT_PATH"
env | LC_ALL=C sort | sed -n '/^TOKENHSI_/p; /^CUDA_VISIBLE_DEVICES=/p; /^CUDA_MPS_/p; /^HOLD_AT_/p; /^SEED=/p; /^CHECK=/p; /^MAX_ITERATIONS=/p; /^RESUME_CHECKPOINT=/p' > "$OUTPUT_PATH/launch.env"
cp tokenhsi/data/cfg/multi_agent/table_hold_at_full_ft.yaml "$OUTPUT_PATH/env.yaml"
cp tokenhsi/data/cfg/train/rlg/table_hold_at_full_ft.yaml "$OUTPUT_PATH/train.yaml"
set -- --task_graph random_scenario --seed "${SEED:-42}"
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python -u tokenhsi/run.py --task HumanoidTableHoldAtFT \
 --cfg_train "$OUTPUT_PATH/train.yaml" --cfg_env "$OUTPUT_PATH/env.yaml" \
 --motion_file tokenhsi/data/dataset_table_hold_at.yaml \
 --num_agents 1 --num_envs 2048 --num_objects 1 \
 --output_path "$OUTPUT_PATH" --experiment TableHoldAtFullFT --no_video "$@"
