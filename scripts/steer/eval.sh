#!/bin/bash
# Evaluate a trained steer-carry checkpoint.
#   STEER_CKPT=output/steer/<tag>/<run>/nn/Humanoid.pth bash scripts/steer/eval.sh [num_envs]
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer

ENVS=${1:-512}
CKPT=${STEER_CKPT:?STEER_CKPT is required}
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

python ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint "$CKPT" \
    --test --eval --eval_task carry \
    --num_envs "$ENVS" --headless
