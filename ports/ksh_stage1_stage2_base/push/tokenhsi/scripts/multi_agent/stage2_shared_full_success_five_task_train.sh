#!/bin/bash
# [num_agents=4] [num_envs=1024] [num_objects=12]
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
. "$(dirname "$0")/runtime_env.sh"
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -eu
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_shared_full_success_five_task train
set -- --num_agents "${1:-4}" --num_envs "${2:-1024}" --num_objects "${3:-12}" --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "${STAGE1_CHECKPOINT:-stage1/Stage1V2FixedFiveTask.pth}"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
python tokenhsi/run.py --task HumanoidMAFiveTaskStage2 \
  --cfg_env "${SHARED_SUPPORT_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_shared_full_success_five_task.yaml}" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_shared_full_success_five_task.yaml \
  --motion_file tokenhsi/data/dataset_five_task.yaml \
  --output_path "$OUTPUT_PATH" --experiment Stage2SharedFullSuccessFiveTask --seed "${SEED:-42}" --headless --no_video "$@"
