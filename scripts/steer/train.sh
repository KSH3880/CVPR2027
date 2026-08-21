#!/bin/bash
# Fine-tune the steer-carry adapt scaffold.
#   STEER_LR=2e-5 STEER_ITERS=400 bash scripts/steer/train.sh [num_envs]
# STEER_ABLATE=2 masks the steering token out of attention (adaptation control).
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer

ENVS=${1:-2048}
LR=${STEER_LR:-2e-5}
ITERS=${STEER_ITERS:-400}
TAG=${STEER_TAG:-train}

GEN=/home/hwanhee/CVPR2027/runs/gen_cfgs/steer
mkdir -p "$GEN"
CFG=$GEN/${TAG}.yaml
BASECFG=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml
# the paper uses disc_reward_scale 3 for its object-shape adaptations ("for better
# motion naturalness"); everything else in that cfg is byte-identical
[ -n "${STEER_DISC3:-}" ] && BASECFG=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_disc3.yaml
sed "s/^\( *\)learning_rate: .*/\1learning_rate: ${LR}/" "$BASECFG" > "$CFG"

RESUME=""
if [ -n "${STEER_RESUME:-}" ]; then
    [ -f "$STEER_RESUME" ] || { echo "no such checkpoint: $STEER_RESUME"; exit 1; }
    RESUME="--resume 1 --checkpoint $STEER_RESUME"
fi

python ./tokenhsi/run.py --task HumanoidSteerCarry $RESUME \
    --cfg_train "$CFG" \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs "$ENVS" --max_iterations "$ITERS" \
    --output_path output/steer/${TAG} \
    --headless
