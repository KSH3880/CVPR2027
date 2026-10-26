#!/bin/bash
# <Stage2 checkpoint> [num_agents=4] [num_envs=1] [num_objects=8] [repeats=10]
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo 'Provide an existing Stage2 checkpoint.pth' >&2
  exit 1
fi
CKPT=$1
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_before_ground_mix226_actorbias0_1200 eval
ENV_CONFIG=${BEFORE_GROUND_MIX226_ACTORBIAS0_1200_ENV_CONFIG:-$(dirname "$(dirname "$CKPT")")/relation_config.yaml}
if [ ! -f "$ENV_CONFIG" ]; then
  echo 'Saved relation_config.yaml is missing; set BEFORE_GROUND_MIX226_ACTORBIAS0_1200_ENV_CONFIG explicitly' >&2
  exit 1
fi
set -- --num_agents "${2:-4}" --num_envs "${3:-1}" --num_objects "${4:-8}" --eval_repeats "${5:-10}" --task_graph "${TASK_GRAPH:-random_scenario}"
case "${BEFORE_FOLLOWER_NEAR:-0}" in
  0) ;;
  1) set -- "$@" --before_follower_near ;;
  2) set -- "$@" --before_follower_short ;;
  *) echo 'BEFORE_FOLLOWER_NEAR must be 0 (mixture), 1 (near), or 2 (short)' >&2; exit 1 ;;
esac
if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
python tokenhsi/run.py --task HumanoidMABefore --cfg_env "$ENV_CONFIG" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_actorbias0_1200.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "$OUTPUT_PATH" --test --eval --no_video --seed "${SEED:-42}" "$@"
