#!/bin/bash
# S0 probe: no training. Overwrites the carry target with a lookahead point on a
# ground-truth path and scores against the real placement target.
#   STEER_LOOKAHEAD=1.2 bash scripts/steer/s0.sh [num_envs]
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer

ENVS=${1:-512}
export STEER_OUT=${STEER_OUT:-/home/hwanhee/CVPR2027/runs/steer}
export STEER_TAG=${STEER_TAG:-probe_la${STEER_LOOKAHEAD:-1.2}}

python ./tokenhsi/run.py --task HumanoidSteerProbe \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint output/tokenhsi/ckpt_stage1.pth \
    --test --eval --eval_task carry \
    --num_envs "$ENVS" --headless
