#!/bin/bash
# Evaluate one F-series checkpoint on the whole-task path.
#   STEER_CKPT=... STEER_TAG=f1_4_D bash scripts/steer/f1_eval.sh [num_envs]
#
# The env is configured from STEER_* exactly as in training, so an eval must be
# handed the same path settings the policy was trained under. Scoring the
# approach leg also needs the trajectory log, hence STEER_LOG.
#
# cfg_env is _cp32.yaml -- the SAME file training uses. It used to point at
# amp_humanoid_steer_carry.yaml, which was left on the terrain-era settings when
# the training config was cleaned up on 08-15. Every F8-F12 number before
# 08-16 09:00 was therefore scored under rules the policy never trained on:
# randomHeightProb 1.0 + minBottomSurfaceHeight 0.5 (box always on a platform
# at least 0.5 m up, vs 50 % on the ground in training), contactBodies including
# shins and thighs (episodes terminated on contacts training never penalised),
# and eval.skill split 50/50 with loco_terrain. Rankings survive -- every arm
# was scored the same way -- but the absolute success rates do not.
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

ENVS=${1:-512}
CKPT=${STEER_CKPT:?STEER_CKPT is required}
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

export STEER_CURVATURE=${STEER_CURVATURE:-0.15}
export STEER_K=${STEER_K:-6}
export STEER_SPACING=${STEER_SPACING:-0.4}
export STEER_APPROACH=${STEER_APPROACH:-1}
export STEER_LOG=${STEER_LOG:-900}

python -u ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry_cp32.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint "$CKPT" \
    --test --eval --eval_task carry \
    --num_envs "$ENVS" --headless
