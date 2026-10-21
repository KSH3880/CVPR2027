#!/bin/bash
set -eu
if [ "${TOKENHSI_GPU:-}" != 6 ]; then echo 'Use TOKENHSI_GPU=6'; exit 1; fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/validate_bones_door_amp.py \
  --task HumanoidMADoor \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_bones_left_amp.yaml \
  --motion_file tokenhsi/data/dataset_door_bones_left.yaml \
  --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video \
  --test --eval --task_graph open --num_threads 1 \
  --output_path "${OUTPUT_PATH:-output/door_bones_left_amp_check}"
