#!/bin/bash
. "$(dirname "$0")/runtime_env.sh"
set -eu
PUSH_AGENTS=${1:-2}
PUSH_ENVS=${2:-2048}
PUSH_OBJECTS=${3:-4}
PUSH_EXPERIMENT=${STAGE1_PUSH_EXPERIMENT:-stage1_v2_fixed_push}
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output "$PUSH_EXPERIMENT" train
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- --checkpoint "$RESUME_CHECKPOINT" --output-dir "$OUTPUT_PATH"
else
  set -- --config "${STAGE1_PUSH_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/$PUSH_EXPERIMENT.yaml}" --output-dir "$OUTPUT_PATH"
fi
case "$PUSH_EXPERIMENT" in
  stage1_v2_fixed_push_360) set -- "$@" --require-navigation ;;
  stage1_v2_fixed_push_360_raw|stage1_v2_fixed_push_360_raw_cuboid) set -- "$@" --require-navigation --require-raw ;;
esac
if [ -n "${PUSH_CHECKPOINT:-}" ]; then set -- "$@" --warm-start "$PUSH_CHECKPOINT"; fi
if [ -n "${STAGE1_CHECKPOINT:-}" ]; then set -- "$@" --source "$STAGE1_CHECKPOINT"; fi
if [ -n "${OLD_TASK_FRACTION:-}" ]; then set -- "$@" --old-fraction "$OLD_TASK_FRACTION"; fi
if [ -n "${TEACHER_KL_COEF:-}" ]; then set -- "$@" --teacher-kl "$TEACHER_KL_COEF"; fi
ENV_CONFIG=$(python tokenhsi/utils/stage1_push.py "$@")
set -- --task_graph random_scenario --seed "${SEED:-42}" --task_camera "${TASK_CAMERA:-stack}"
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${MINIBATCH_SIZE:-}" ]; then set -- "$@" --minibatch_size "$MINIBATCH_SIZE"; fi
python tokenhsi/run.py --task HumanoidMAStage1Push \
  --cfg_train "tokenhsi/data/cfg/train/rlg/$PUSH_EXPERIMENT.yaml" --cfg_env "$ENV_CONFIG" \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml --num_agents "$PUSH_AGENTS" \
  --num_envs "$PUSH_ENVS" --num_objects "$PUSH_OBJECTS" --output_path "$OUTPUT_PATH" \
  --experiment "${STAGE1_PUSH_RUN_NAME:-Stage1V2FixedPush}" --no_video "$@"
