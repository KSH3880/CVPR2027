#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/validate_open_push_scene.py \
  --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/open_push_push180_random_scene.yaml \
  --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
  --num_agents 2 --num_envs 2 --num_objects 4 --task_graph random_open_push \
  --test --eval --headless --no_video --output_path "${OUTPUT_PATH:-output/open_push_random_scene_sim_check}"
