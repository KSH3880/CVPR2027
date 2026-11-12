#!/bin/bash
set -eu
METHOD=${1:?initial|tokenhsi|fsm|ours}
SEARCH_SEED=${2:-0}
source "$(dirname "$0")/../multi_agent/runtime_env.sh"
export PUSH_FIXED_PROFILE=${PUSH_FIXED_PROFILE:-reference}
if [ "$PUSH_FIXED_PROFILE" = eval_min ]; then
    export NATURAL_PUSH_ROOT=${NATURAL_PUSH_ROOT:-$TOKENHSI_ROOT/output_etc/check/push_fixed_evalmin_20261113}
else
    export NATURAL_PUSH_ROOT=${NATURAL_PUSH_ROOT:-$TOKENHSI_ROOT/output_etc/check/push_fixed_near_far_goal5_20261112}
fi
export COMPARE_SOURCE_ROOT=$NATURAL_PUSH_ROOT/source
export COMPARE_OUTPUT_ROOT=$NATURAL_PUSH_ROOT/rollouts
export COMPARE_METHOD=$METHOD
export PUSH_STAGE_PROFILE=sideways2
if [ "$METHOD" = queue ]; then
    export NATURAL_PUSH_WRAPPER=tokenhsi/scripts/analysis/push_fixed_search.sh
    export NATURAL_PUSH_WORKERS=${NATURAL_PUSH_WORKERS:-4}
    export NATURAL_PUSH_CONTINUE=1 NATURAL_PUSH_START_SEED=$SEARCH_SEED
    export NATURAL_PUSH_NUM_ENVS=${COMPARE_NUM_ENVS:-1024}
    exec python tokenhsi/scripts/analysis/natural_push_queue.py
fi
if [ "$METHOD" = initial ]; then
    export PUSH_PREFILTER_ONLY=1
    DESTINATION=$COMPARE_SOURCE_ROOT/seed${SEARCH_SEED}
else
    case "$METHOD" in
        tokenhsi) export COMPARE_SELECTION_FILE=$COMPARE_SOURCE_ROOT/seed${SEARCH_SEED}/initial_prefilter.json ;;
        fsm) export COMPARE_SELECTION_FILE=$COMPARE_OUTPUT_ROOT/seed${SEARCH_SEED}/tokenhsi/selection.json ;;
        ours) export COMPARE_SELECTION_FILE=$COMPARE_OUTPUT_ROOT/seed${SEARCH_SEED}/fsm/selection.json ;;
        *) exit 2 ;;
    esac
    DESTINATION=$COMPARE_OUTPUT_ROOT/seed${SEARCH_SEED}/$METHOD
fi
exec python tokenhsi/scripts/analysis/push_fixed_rollout.py \
 --task HumanoidMACoupledPush --cfg_train tokenhsi/data/cfg/train/rlg/stage2_distill_900.yaml --cfg_env '' \
 --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
 --checkpoint /home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth \
 --num_agents 2 --num_envs "${COMPARE_NUM_ENVS:-1024}" --num_objects 4 --seed "$SEARCH_SEED" \
 --task_graph all_coupled --coupled_assignment ordered --eval_skills loco --eval_skill_probs 1.0 --eval_repeats 1 \
 --test --eval --headless --no_video --output_path "$DESTINATION"
