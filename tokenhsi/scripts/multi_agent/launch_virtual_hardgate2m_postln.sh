#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODE=$1
shift
. "$SCRIPT_DIR/runtime_env.sh"
NAME=stage2_before_virtual_hardgate2m_postln
if [ "$MODE" = vnc ]; then
  export HEADLESS=0 PORT=${PORT:-6142}
  exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/${NAME}_test.sh" "$@"
fi
. "$SCRIPT_DIR/output_paths.sh"
tokenhsi_select_output "$NAME" "$MODE"
OUTPUT_PATH=$(realpath -m "$OUTPUT_PATH")
CFG_ROOT=$TOKENHSI_ROOT/tokenhsi/data/cfg
if [ "$MODE" = train ]; then
  GRAPH=random_scenario
  CFG_ENV=$CFG_ROOT/multi_agent/$NAME.yaml
  set -- --num_agents "${1:-4}" --num_envs "${2:-1024}" --num_objects "${3:-8}"
  if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    set -- "$@" --checkpoint "$(realpath "$RESUME_CHECKPOINT")" --resume 1
  else
    set -- "$@" --transfer_checkpoint "$(realpath "${STAGE1_CHECKPOINT:-checkpoints/Stage1Push_00026000.pth}")"
  fi
  if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
  set -- "$@" --headless
else
  GRAPH=${TASK_GRAPH:-random_scenario}
  CFG_ENV=${ENV_CONFIG:-}
  CKPT=$(realpath "$1")
  set -- --checkpoint "$CKPT" --num_agents "${2:-4}" --num_envs "${3:-1}" --num_objects "${4:-8}" --eval_repeats "${5:-10}" --test --eval
  if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
  set -- "$@" --episode_length "${EPISODE_LENGTH:-1200}"
  if [ -n "${EVAL_SKILLS:-}" ]; then
    set -- "$@" --eval_skills "$EVAL_SKILLS" --eval_skill_probs "${EVAL_SKILL_PROBS:?Set EVAL_SKILL_PROBS with EVAL_SKILLS}"
  fi
fi
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$TOKENHSI_ROOT/output_etc/check/virtual_hardgate2m_postln_20261103/torch_extensions}
cd "$TOKENHSI_ROOT/experiments/$NAME"
exec python tokenhsi/run.py --task HumanoidMABeforePush \
  --cfg_env "$CFG_ENV" --cfg_train "$CFG_ROOT/train/rlg/$NAME.yaml" \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --output_path "$OUTPUT_PATH" --task_graph "$GRAPH" \
  --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}" --no_video "$@"
