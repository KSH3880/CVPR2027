#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_push_actuator_load.py \
 --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction1_random_goal.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml --num_agents 2 --num_envs 864 --num_objects 4 \
 --task_graph push --test --eval --headless --no_video \
 --output_path "${OUTPUT_PATH:-output/push_actuator_load_20261012}"
