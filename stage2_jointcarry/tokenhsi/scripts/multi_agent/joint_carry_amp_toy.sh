#!/bin/bash
. "$(dirname "$0")/runtime_env.sh"
set -eu
MODE=${1:?train, eval or reward_eval}
AMP_WEIGHT=${2:?0.5 or 0}
CKPT=${3:?checkpoint.pth}
OUT=${4:?output directory}
UPDATES=${5:-100}
RUNNER=tokenhsi/tests/toy_joint_carry_amp.py
case "$MODE" in
  train) NUM_ENVS=2048; LENGTH=600; FLAGS=(--resume 1) ;;
  eval) NUM_ENVS=64; LENGTH=180; FLAGS=(--test --eval --eval_skills carryWith --eval_skill_probs 1.0 --eval_repeats 1) ;;
  reward_eval) RUNNER=tokenhsi/tests/toy_joint_carry_goal_reward.py; NUM_ENVS=64; LENGTH=180; FLAGS=(--test --eval --eval_skills carryWith --eval_skill_probs 1.0 --eval_repeats 1) ;;
  *) echo 'mode must be train, eval or reward_eval' >&2; exit 1 ;;
esac
python tokenhsi/scripts/prepare_joint_carry.py
python "$RUNNER" \
  --toy-amp-weight "$AMP_WEIGHT" --toy-updates "$UPDATES" \
  --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --checkpoint "$CKPT" --num_agents 2 --num_envs "$NUM_ENVS" --num_objects 4 \
  --episode_length "$LENGTH" --output_path "$OUT" --experiment JointCarryAmpToy \
  --headless --no_video --task_graph random_scenario --seed 42 "${FLAGS[@]}"
