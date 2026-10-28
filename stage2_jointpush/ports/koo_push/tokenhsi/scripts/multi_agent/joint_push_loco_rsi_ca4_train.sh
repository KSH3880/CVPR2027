#!/bin/bash
set -eu
export TOKENHSI_GPU=6
. "$(dirname "$0")/runtime_env.sh"
PROJECT_ROOT=$(cd "$TOKENHSI_ROOT/../../.." && pwd)
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$PROJECT_ROOT/runs/jointpush/push_stage1_repro_20261012/checkpoints/OpenPushPush180ContactAtWarmstart_00029000.pth}
TRAIN_OUTPUT=${OUTPUT_PATH:-$PROJECT_ROOT/jointpush/output/training}
CFG_ENV=tokenhsi/data/cfg/multi_agent/joint_push_loco_rsi_ca4.yaml
set -- --task_graph push
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  CFG_ENV=$(python tokenhsi/scripts/multi_agent/joint_push_checkpoint_env.py "$RESUME_CHECKPOINT" "$CFG_ENV" "$TRAIN_OUTPUT/configs")
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
exec python tokenhsi/run.py --task HumanoidMAJointPush \
 --cfg_env "$CFG_ENV" \
 --cfg_train tokenhsi/data/cfg/train/rlg/joint_push_loco_rsi_ca4.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
 --num_agents 2 --num_envs 2048 --num_objects 4 \
 --output_path "$TRAIN_OUTPUT" \
 --experiment JointPushLocoRsiCa4 --headless --no_video "$@"
