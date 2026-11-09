#!/bin/bash

python ./tokenhsi/run.py --task HumanoidCompClimbCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_comp.yaml \
    --cfg_env tokenhsi/data/cfg/comp_interaction_skills/amp_humanoid_comp_climb_carry.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint output/tokenhsi/ckpt_stage2_comp_climb_carry.pth \
    --test \
    --num_envs 512 \
    --headless \
    --eval \
    --seed 0