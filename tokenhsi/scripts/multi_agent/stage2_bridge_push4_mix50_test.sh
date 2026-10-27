#!/bin/bash
# <Stage2 checkpoint> [num_agents=4] [num_envs=1] [num_objects=8] [repeats=10]
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_sm120}
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo 'Provide an existing Stage2 checkpoint.pth' >&2
  exit 1
fi
CKPT=$1
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_bridge_push4_mix50 eval
ENV_CONFIG=${BRIDGE_PUSH4_MIX50_ENV_CONFIG:-$(dirname "$(dirname "$CKPT")")/relation_config.yaml}
if [ ! -f "$ENV_CONFIG" ]; then
  echo 'Saved relation_config.yaml is missing; set BRIDGE_PUSH4_MIX50_ENV_CONFIG explicitly' >&2
  exit 1
fi
set -- --num_agents "${2:-4}" --num_envs "${3:-1}" --num_objects "${4:-8}" --eval_repeats "${5:-10}" --task_graph "${TASK_GRAPH:-random_scenario}"
if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
python tokenhsi/run.py --task HumanoidMABridgePush4 --cfg_env "$ENV_CONFIG" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_bridge_push4_mix50.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml --checkpoint "$CKPT" \
  --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "$OUTPUT_PATH" --test --eval --no_video --seed "${SEED:-42}" "$@"
