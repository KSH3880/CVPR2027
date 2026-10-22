#!/bin/bash
# No policy/checkpoint; CPU affinity and all compute thread pools are limited.
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
AUDIT_SIM=${AUDIT_SIM:-gpu}
DOOR_EXPERIMENT=${DOOR_EXPERIMENT:-door_open_unified_approach_amp}
set -- --task HumanoidMADoor \
  --cfg_env "tokenhsi/data/cfg/multi_agent/$DOOR_EXPERIMENT.yaml" \
  --cfg_train "tokenhsi/data/cfg/train/rlg/$DOOR_EXPERIMENT.yaml" \
  --motion_file tokenhsi/data/dataset_door_loco.yaml \
  --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video \
  --test --eval --task_graph open --num_threads 1 \
  --output_path "${OUTPUT_PATH:-output/door_physics_check_$AUDIT_SIM}"
case "$AUDIT_SIM" in
  cpu) set -- "$@" --sim_device cpu --pipeline cpu --rl_device cpu ;;
  gpu) ;;
  *) echo "AUDIT_SIM must be gpu or cpu" >&2; exit 1 ;;
esac
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/validate_door_physics.py "$@"
