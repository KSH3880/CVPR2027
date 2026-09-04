#!/bin/bash
# S1 wiring check: Carry + steering window token on the adapt scaffold.
# STEER_ZERO=1 emits a zero window, which must reproduce pretrained carry.
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer

ENVS=${1:-512}
python ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --test --eval --eval_task carry \
    --num_envs "$ENVS" --headless
