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
    SOURCE=${STAGE1_CHECKPOINT:-/home/hwanhee/koo_cvpr/stage1_fix/output/open_push_push180_contact_at_warmstart}
    SOURCE=$(python tokenhsi/scripts/multi_agent/partition_push_setup.py checkpoint "$SOURCE")
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
exec python tokenhsi/run.py --task HumanoidMAPartitionPush \
 --cfg_env "$CFG_ENV" --cfg_train "$CFG_TRAIN" \
 --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
 --num_agents 4 --num_envs "${NUM_ENVS:-1024}" --num_objects 8 \
 --output_path "$OUT" --experiment PartitionPushCa4 --no_video "$@"
