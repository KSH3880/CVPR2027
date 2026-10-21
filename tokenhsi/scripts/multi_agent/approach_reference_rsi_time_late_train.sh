#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 6|7) ;; *) echo "Only GPUs 6 and 7 are allowed" >&2; exit 2;; esac
RSI_PHYSICAL_GPU=$TOKENHSI_GPU
export CUDA_VISIBLE_DEVICES=$TOKENHSI_GPU
. "$(dirname "$0")/runtime_env.sh"
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_reference_rsi_time_late.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_agents 2 --num_envs 2048 --num_objects 4 \
  --output_path "${OUTPUT_PATH:-../runs/reference_rsi/time_late_train}" \
  --experiment ReferenceRSITimeLate --seed "${SEED:-0}" \
  --max_iterations "${MAX_ITERATIONS:-3000}" --task_graph random_scenario --headless --no_video
