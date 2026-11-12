#!/bin/bash
set -eu
export COMPARE_METHOD=${1:?method: ours/tokenhsi/fsm}
COMPARE_SEED=${2:-2}
source "$(dirname "$0")/../multi_agent/runtime_env.sh"
exec python tokenhsi/scripts/analysis/push_comparison_rollout.py \
 --task HumanoidMACoupledPush --cfg_train tokenhsi/data/cfg/train/rlg/stage2_distill_900.yaml --cfg_env '' \
 --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
 --checkpoint /home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth \
 --num_agents 2 --num_envs "${COMPARE_NUM_ENVS:-256}" --num_objects 4 --seed "$COMPARE_SEED" --task_graph all_coupled \
 --coupled_assignment ordered --eval_skills loco --eval_skill_probs 1.0 --eval_repeats 1 \
 --test --eval --headless --no_video \
 --output_path "${COMPARE_OUTPUT_ROOT:-output_etc/check/push_comparison_capture_20261112}/seed${COMPARE_SEED}/${COMPARE_METHOD}"
