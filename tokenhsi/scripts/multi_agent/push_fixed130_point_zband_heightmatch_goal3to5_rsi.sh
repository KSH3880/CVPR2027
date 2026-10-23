#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
run_height_check() {
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python "$1" \
 --task HumanoidMAOpenPushPoint --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5.yaml \
 --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 64 --num_objects 4 \
 --task_graph push --test --eval --headless --no_video --num_threads 1 \
 --output_path "${OUTPUT_PATH:-output/push_fixed130_point_zband_heightmatch_goal3to5_rsi_check}"

}
run_height_check tokenhsi/scripts/multi_agent/build_push_height_matched_rsi.py
run_height_check tokenhsi/scripts/multi_agent/check_push_anchor_rsi.py
