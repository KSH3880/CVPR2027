#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
NUM_ENVS=${CAPACITY_NUM_ENVS:-220}
CHECKPOINT=${1:-output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00029000.pth}
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_open_push_policy_capacity.py \
 --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint "$CHECKPOINT" \
 --num_agents 2 --num_envs "$NUM_ENVS" --num_objects 4 --task_graph push \
 --test --eval --headless --no_video --seed 42 --eval_skills loco --eval_skill_probs 1.0 \
 --output_path "${OUTPUT_PATH:-output/push_policy_mass_friction_grid_20261012}"
