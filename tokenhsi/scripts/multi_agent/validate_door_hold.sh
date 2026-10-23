#!/bin/bash
set -eu
if [ "${TOKENHSI_GPU:-}" != 6 ]; then echo 'Use TOKENHSI_GPU=6'; exit 1; fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
set -- --task HumanoidMADoorHold \
 --cfg_env tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/door_open_hold_amp.yaml \
 --motion_file tokenhsi/data/dataset_door_bones_left.yaml \
 --num_envs 2 --num_agents 1 --num_objects 1 --headless --no_video \
 --test --eval --task_graph open --num_threads 1 \
 --output_path "${OUTPUT_PATH:-output/door_hold_check_${AUDIT_SIM:-gpu}}"
if [ "${AUDIT_SIM:-gpu}" = cpu ]; then set -- "$@" --sim_device cpu --pipeline cpu --rl_device cpu; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/validate_door_hold.py "$@"
