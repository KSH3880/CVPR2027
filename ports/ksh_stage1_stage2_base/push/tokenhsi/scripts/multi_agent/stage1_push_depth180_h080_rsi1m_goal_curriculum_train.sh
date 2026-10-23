#!/bin/bash
# Isolated imported Stage1: 2 humans, 2048 environments, 4 object inputs.
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
. "$(dirname "$0")/runtime_env.sh"
set -eu
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage1_push_depth180_h080_rsi1m_goal_curriculum train
set -- --num_agents "${1:-2}" --num_envs "${2:-2048}" --num_objects "${3:-4}" --task_graph random_scenario
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${TRANSFER_CHECKPOINT:-}" ]; then echo 'PUSH requires scratch or its own resume checkpoint.' >&2; exit 2; fi
python tokenhsi/run.py --task HumanoidMAStage1Push \
 --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml \
 --motion_file tokenhsi/data/dataset_stage1_push.yaml --experiment Stage1PushDepth180H080Rsi1mGoalCurriculum \
 --output_path "$OUTPUT_PATH" --seed "${SEED:-42}" --headless --no_video "$@"
