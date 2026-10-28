#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage1_jointpush_density3_cone90_finetune train
set -- --num_agents "${1:-2}" --num_envs "${2:-2048}" --num_objects "${3:-4}"
set -- "$@" --task_graph random_scenario --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}" --no_video
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${MINIBATCH_SIZE:-}" ]; then set -- "$@" --minibatch_size "$MINIBATCH_SIZE"; fi
if [ -n "${STAGE1_CHECKPOINT:-}" ]; then set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"; fi
python tokenhsi/run.py --task HumanoidMAStage1JointPush \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_jointpush_density3_cone90_finetune.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_jointpush_density3_cone90_finetune.yaml \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --output_path "$OUTPUT_PATH" "$@"
