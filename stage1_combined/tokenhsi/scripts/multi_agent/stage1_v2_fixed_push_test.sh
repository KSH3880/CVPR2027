#!/bin/bash
. "$(dirname "$0")/runtime_env.sh"
set -eu
CKPT=${1:?Provide a PUSH checkpoint}
PUSH_AGENTS=${2:-2}
PUSH_ENVS=${3:-1}
PUSH_OBJECTS=${4:-4}
PUSH_REPEATS=${5:-3}
PUSH_EPISODE_LENGTH=${6:-${EPISODE_LENGTH:-}}
PUSH_EXPERIMENT=${STAGE1_PUSH_EXPERIMENT:-stage1_v2_fixed_push}
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output "$PUSH_EXPERIMENT" eval
set -- --checkpoint "$CKPT" --output-dir "$OUTPUT_PATH"
case "$PUSH_EXPERIMENT" in
  stage1_v2_fixed_push_360) set -- "$@" --require-navigation ;;
  stage1_v2_fixed_push_360_raw|stage1_v2_fixed_push_360_raw_cuboid) set -- "$@" --require-navigation --require-raw ;;
esac
ENV_CONFIG=$(python tokenhsi/utils/stage1_push.py "$@")
set -- --task_graph "${TASK_GRAPH:-push}" --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}"
set -- "$@" --push_spawn_mode "${PUSH_SPAWN_MODE:-random}"
set -- "$@" --push_rsi_mode "${PUSH_RSI_MODE:-}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "$PUSH_EPISODE_LENGTH" ]; then set -- "$@" --episode_length "$PUSH_EPISODE_LENGTH"; fi
python tokenhsi/run.py --task HumanoidMAStage1Push \
  --cfg_train "tokenhsi/data/cfg/train/rlg/$PUSH_EXPERIMENT.yaml" --cfg_env "$ENV_CONFIG" \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml --checkpoint "$CKPT" \
  --num_agents "$PUSH_AGENTS" --num_envs "$PUSH_ENVS" --num_objects "$PUSH_OBJECTS" \
  --eval_repeats "$PUSH_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "$OUTPUT_PATH" --test --eval --no_video "$@"
