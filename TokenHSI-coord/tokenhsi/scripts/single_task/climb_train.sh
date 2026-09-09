#!/bin/bash

python ./tokenhsi/run.py --task HumanoidClimb \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_climb.yaml \
    --motion_file tokenhsi/data/dataset_amass_climb/dataset_amass_climb.yaml \
    --num_envs 4096 \
    --headless