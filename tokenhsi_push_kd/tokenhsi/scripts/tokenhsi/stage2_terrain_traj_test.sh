#!/bin/bash

python ./tokenhsi/run.py --task HumanoidAdaptTrajGround2Terrain \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_traj_ground2terrain.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco_stairs/dataset_amass_loco_stairs.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint output/tokenhsi/ckpt_stage2_terrainShape_traj.pth \
    --test \
    --num_envs 1
