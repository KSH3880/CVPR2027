#!/bin/bash
set -eu
export TOKENHSI_GPU=6
. "$(dirname "$0")/runtime_env.sh"
PROJECT_ROOT=$(cd "$TOKENHSI_ROOT/../../.." && pwd)
CKPT=${STAGE1_CHECKPOINT:-$PROJECT_ROOT/runs/jointpush/push_stage1_repro_20261012/checkpoints/OpenPushPush180ContactAtWarmstart_00029000.pth}
set -- --task_graph push
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
exec python tokenhsi/run.py --task HumanoidMAJointPushVirtual \
 --cfg_env "${VIRTUAL_CFG_ENV:-tokenhsi/data/cfg/multi_agent/joint_push_virtual_half_stage1.yaml}" \
 --cfg_train tokenhsi/data/cfg/train/rlg/joint_push_virtual_half_stage1.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
 --checkpoint "$CKPT" --num_agents 2 --num_envs "${NUM_ENVS:-16}" --num_objects 4 \
 --eval_repeats "${REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 --seed "${SEED:-42}" \
 --output_path "${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/virtual_rear_cubes_stage1}" \
 --test --eval --no_video "$@"
