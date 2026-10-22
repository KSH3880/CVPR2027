#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
extra=()
if [ "${RSI_SIM:-cpu}" = cpu ]; then extra+=(--sim_device cpu --pipeline cpu --rl_device cpu); fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_push_random_scene.py \
 --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/push_toy_contact_align_at.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_toy_contact_align_at.yaml --motion_file tokenhsi/data/dataset_open_push.yaml \
 --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video --test --eval \
 --task_graph random_open_push --num_threads 1 --output_path "${OUTPUT_PATH:-output/push_toy_contact_align_at_connections}" "${extra[@]}"
