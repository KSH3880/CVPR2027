#!/bin/bash

python ./tokenhsi/run.py --task HumanoidSit \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_sit.yaml \
    --motion_file tokenhsi/data/dataset_sit/dataset_sit.yaml \
    --num_envs 4096 \
    --headless
