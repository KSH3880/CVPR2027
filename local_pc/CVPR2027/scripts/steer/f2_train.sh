#!/bin/bash
# F2: one steer-carry slot at the settled 4096-env configuration.
#   STEER_TAG=f2_3_D STEER_GPU=0 STEER_CFG=f2_base bash scripts/steer/f2_train.sh
#
# Everything env-scaling-related is pinned here rather than left to the caller,
# because the four settings only make sense together: 4096 envs needs a 32M
# contact-pair buffer to stop PhysX dropping contacts, needs minibatch 32768 to
# keep the number of sequential updates per rollout at the 2048 reference, and
# needs spawns spread so the broadphase pair count does not blow up. See
# docs/ENV_SCALING.md.
#
# A per-batch copy, never edited while jobs are running: bash reads a script
# incrementally, so editing one under a live job truncates it mid-parse.
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

TAG=${STEER_TAG:?STEER_TAG is required}
CFG=${STEER_CFG:-f2_base}
ITERS=${STEER_ITERS:-3000}
ENVS=${STEER_ENVS:-4096}
export STEER_CELL=${STEER_CELL:-60}

python -u ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train /home/hwanhee/CVPR2027/runs/gen_cfgs/steer/${CFG}.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry_cp32.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs "$ENVS" --max_iterations "$ITERS" \
    --output_path output/steer/${TAG} \
    --headless
