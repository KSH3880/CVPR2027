#!/bin/bash
# [num_agents=4] [num_envs=1024] [num_objects=8]
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_before_ground train
set -- --num_agents "${1:-4}" --num_envs "${2:-1024}" --num_objects "${3:-8}" --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "${STAGE1_CHECKPOINT:-stage1/Stage1V2Fixed_00030500.pth}"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
python tokenhsi/run.py --task HumanoidMABefore \
  --cfg_env "${BEFORE_GROUND_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_before_ground.yaml}" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_before_ground.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --output_path "$OUTPUT_PATH" --experiment Stage2BeforeGround --seed "${SEED:-42}" --headless --no_video "$@"
