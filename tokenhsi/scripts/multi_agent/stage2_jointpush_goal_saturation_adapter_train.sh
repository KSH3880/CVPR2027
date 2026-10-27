#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh}
. "$(dirname "$0")/runtime_env.sh"
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_jointpush_goal_saturation_adapter train
set -- --num_agents "${1:-4}" --num_envs "${2:-1024}" --num_objects "${3:-8}"
set -- "$@" --task_graph random_scenario --task_camera "${TASK_CAMERA:-stack}" --seed "${SEED:-42}" --no_video
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "${STAGE1_CHECKPOINT:-pth/Stage1PushFinetune_00010500.pth}"
fi
if [ -n "${MINIBATCH_SIZE:-}" ]; then set -- "$@" --minibatch_size "$MINIBATCH_SIZE"; fi
python "${JOINTPUSH_ENTRYPOINT:-tokenhsi/run.py}" --task HumanoidMAStage2JointPush \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2_jointpush_goal_saturation_adapter.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_jointpush_goal_saturation_adapter.yaml \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --output_path "$OUTPUT_PATH" "$@"
