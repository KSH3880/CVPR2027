#!/bin/bash
# Shared entry point behind the experiments' train/test/VNC wrappers.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXPERIMENT=$1
MODE=$2
shift 2
case "$EXPERIMENT" in
  stage2_coupled_push|stage2_coupled_push_region_goals) TASK=HumanoidMACoupledPush; AGENTS=4; ENVS=1024; OBJECTS=8 ;;
  stage2_before_push|stage2_before_push_near50) TASK=HumanoidMABeforePush; AGENTS=4; ENVS=1024; OBJECTS=8 ;;
  stage1_push) TASK=HumanoidMAPush; AGENTS=2; ENVS=2048; OBJECTS=4 ;;
  stage1) TASK=HumanoidMACarry; AGENTS=2; ENVS=2048; OBJECTS=4 ;;
  stage2_before_nonemask_rsi433_near50|stage2_before|stage2_before_rsi433)
    TASK=HumanoidMABefore; AGENTS=4; ENVS=1024; OBJECTS=8 ;;
  *) echo "Unsupported experiment: $EXPERIMENT" >&2; exit 1 ;;
esac
HORIZON=900
case "$EXPERIMENT" in stage2_*) HORIZON=1200 ;; esac
. "$SCRIPT_DIR/runtime_env.sh"

if [ "$MODE" = vnc ]; then
  export HEADLESS=0
  export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
  bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/launch.sh" "$EXPERIMENT" eval "$@"
  exit $?
fi
. "$SCRIPT_DIR/output_paths.sh"
tokenhsi_select_output "$EXPERIMENT" "$MODE"
if [ "$MODE" = train ]; then
  set -- --num_agents "${1:-$AGENTS}" --num_envs "${2:-$ENVS}" --num_objects "${3:-$OBJECTS}"
  if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
  elif { [ "$TASK" = HumanoidMABefore ] || [ "$TASK" = HumanoidMABeforePush ] || [ "$TASK" = HumanoidMACoupledPush ]; } && [ -n "${STAGE1_CHECKPOINT:-}" ]; then
    set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
  fi
  if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
  set -- "$@" --task_graph random_scenario --headless
elif [ "$MODE" = eval ]; then
  if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo 'Provide an existing checkpoint.pth; paths are relative to the repository root.' >&2
    exit 1
  fi
  CKPT=$1
  set -- --checkpoint "$CKPT" --num_agents "${2:-$AGENTS}" --num_envs "${3:-1}" \
    --num_objects "${4:-$OBJECTS}" --eval_repeats "${5:-10}" \
    --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}" \
    --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" --test --eval
  if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
  if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
  set -- "$@" --episode_length "${EPISODE_LENGTH:-$HORIZON}"
  if { [ "$TASK" = HumanoidMABefore ] || [ "$TASK" = HumanoidMABeforePush ]; }; then
    set -- "$@" --before_pickup_carry_rsi "${BEFORE_PICKUP_CARRY_RSI:-loco}"
    case "${BEFORE_FOLLOWER_NEAR:-0}" in
      0) ;;
      1) set -- "$@" --before_follower_near ;;
      2) set -- "$@" --before_follower_short ;;
      *) echo 'BEFORE_FOLLOWER_NEAR must be 0, 1 or 2 (mix226 only).' >&2; exit 1 ;;
    esac
  fi
else
  echo "Unsupported mode: $MODE" >&2
  exit 1
fi
MOTION=tokenhsi/data/dataset_loco_sit_carry_climb.yaml
if [ "$EXPERIMENT" = stage1_push ] || [ "$EXPERIMENT" = stage2_before_push ] || [ "$EXPERIMENT" = stage2_before_push_near50 ] || [ "$TASK" = HumanoidMACoupledPush ]; then MOTION=tokenhsi/data/dataset_stage1_jointpush.yaml; fi
python tokenhsi/run.py --task "$TASK" \
  --cfg_train "tokenhsi/data/cfg/train/rlg/$EXPERIMENT.yaml" \
  --cfg_env "${ENV_CONFIG:-}" \
  --motion_file "$MOTION" \
  --output_path "$OUTPUT_PATH" --seed "${SEED:-42}" --no_video "$@"
