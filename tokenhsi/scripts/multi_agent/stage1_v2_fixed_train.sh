#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=4]
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage1_v2_fixed train
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-4}
ENV_CONFIG=$(python tokenhsi/utils/stage1_v2_fixed.py \
  --config "${STAGE1_V2_FIXED_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml}" \
  --output-dir "$OUTPUT_PATH")
set -- --task_graph random_scenario
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
set -- "$@" --seed "${SEED:-42}"
set -- "$@" --task_camera "${TASK_CAMERA:-stack}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${MINIBATCH_SIZE:-}" ]; then set -- "$@" --minibatch_size "$MINIBATCH_SIZE"; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml \
  --cfg_env "$ENV_CONFIG" \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "$OUTPUT_PATH" \
  --experiment Stage1V2Fixed --no_video "$@"
