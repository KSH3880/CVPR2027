#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$SCRIPT_DIR/runtime_env.sh"
extra=()
if [ "${HEADLESS:-0}" = 1 ]; then extra+=(--headless); fi
if [ "${REFERENCE_SIM:-gpu}" = cpu ]; then extra+=(--sim_device cpu --pipeline cpu --rl_device cpu); fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python "$SCRIPT_DIR/view_open_reference_motion.py"   --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/open_toy_reference_close.yaml   --cfg_train tokenhsi/data/cfg/train/rlg/open_toy_reference_close.yaml   --motion_file tokenhsi/data/dataset_open_push.yaml --num_envs 1 --num_agents 2 --num_objects 4   --test --eval --task_graph open --no_video --seed 42 --num_threads 1   --output_path "${OUTPUT_PATH:-output/open_reference_motion_view}" "${extra[@]}"
