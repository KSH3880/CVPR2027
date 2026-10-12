#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
if [ "$TOKENHSI_GPU" != 6 ]; then echo "OPEN/PUSH validation uses GPU 6 only" >&2; exit 1; fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
set -- --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/open_push_unified_amp.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --motion_file tokenhsi/data/dataset_open_push.yaml \
 --num_envs 2 --num_agents 2 --num_objects 4 --headless --no_video --test --eval \
 --task_graph random_open_push --num_threads 1 --output_path "${OUTPUT_PATH:-output/open_push_check}"
if [ "${AUDIT_SIM:-gpu}" = cpu ]; then set -- "$@" --sim_device cpu --pipeline cpu --rl_device cpu; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/validate_open_push.py "$@"
