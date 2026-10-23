#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export RSI_SCREEN_BATCH_SIZE=${RSI_SCREEN_BATCH_SIZE:-8}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/build_open_push_rsi.py \
 --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_goal_face.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
 --num_envs "$((2*RSI_SCREEN_BATCH_SIZE))" --num_agents 2 --num_objects 4 --headless --no_video --test --eval \
 --task_graph random_open_push --num_threads 1 --output_path "${OUTPUT_PATH:-output/open_push_friction06_goal_face_rsi_check}"
