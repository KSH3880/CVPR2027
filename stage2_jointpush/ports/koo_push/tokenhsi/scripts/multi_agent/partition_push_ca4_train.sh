#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
PROJECT_ROOT=$(cd "$TOKENHSI_ROOT/../../.." && pwd)
MODE=${PARTITION_PUSH_MODE:-train}
CFG_ENV=${PARTITION_PUSH_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/partition_push_ca4.yaml}
CFG_TRAIN=tokenhsi/data/cfg/train/rlg/partition_push_ca4.yaml
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/partition_push_$MODE}
set -- --task_graph push
if [ "$MODE" = train ]; then
  if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    CFG_ENV=$(python tokenhsi/scripts/multi_agent/partition_push_setup.py config "$RESUME_CHECKPOINT" "$OUT/configs/partition_push.yaml")
    set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
  else
    SOURCE=$(python tokenhsi/scripts/multi_agent/partition_push_setup.py checkpoint)
    set -- "$@" --transfer_checkpoint "$SOURCE"
  fi
  set -- "$@" --headless
  if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
else
  CKPT=${PARTITION_PUSH_CHECKPOINT:?Provide a PartitionPush checkpoint}
  ABLATION_ARGS=()
  if [ -n "${ABLATION_SIZE:-}" ]; then
    read -r -a ABLATION_VALUES <<< "$ABLATION_SIZE"
    if [ "${#ABLATION_VALUES[@]}" != 3 ]; then echo 'ABLATION_SIZE requires X Y Z in metres' >&2; exit 2; fi
    ABLATION_ARGS+=(--size "${ABLATION_VALUES[@]}")
  fi
  if [ -n "${ABLATION_MASS:-}" ]; then ABLATION_ARGS+=(--mass "$ABLATION_MASS"); fi
  CFG_ENV=$(python tokenhsi/scripts/multi_agent/partition_push_setup.py config "$CKPT" "$OUT/configs/partition_push.yaml" "${ABLATION_ARGS[@]}")
  set -- "$@" --checkpoint "$CKPT" --test --eval --eval_repeats "${REPEATS:-3}" --eval_skills loco --eval_skill_probs 1.0
  if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
exec python tokenhsi/scripts/multi_agent/partition_push_launch.py --task HumanoidMAPartitionPush \
 --cfg_env "$CFG_ENV" --cfg_train "$CFG_TRAIN" \
 --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
 --num_agents 4 --num_envs "${NUM_ENVS:-1024}" --num_objects 8 \
 --output_path "$OUT" --experiment PartitionPushCa4 --no_video "$@"
