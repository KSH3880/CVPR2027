#!/bin/bash
# Non-training CPU PhysX contact-pair audit, using the current OPEN assets/PD.
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_open_reference_push.py \
  --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/open_toy_strong_close.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/open_toy_strong_close.yaml \
  --motion_file tokenhsi/data/dataset_open_push.yaml --num_envs 2 --num_agents 2 --num_objects 4 \
  --headless --no_video --test --eval --task_graph open --num_threads 1 --seed 42 \
  --sim_device cpu --pipeline cpu --rl_device cpu \
  --output_path "${OUTPUT_PATH:-output/open_reference_push_check}"
