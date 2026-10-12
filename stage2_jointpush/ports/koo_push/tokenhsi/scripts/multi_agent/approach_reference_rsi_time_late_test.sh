#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 6|7) ;; *) echo "Only GPUs 6 and 7 are allowed" >&2; exit 2;; esac
RSI_PHYSICAL_GPU=$TOKENHSI_GPU
export CUDA_VISIBLE_DEVICES=$TOKENHSI_GPU
. "$(dirname "$0")/runtime_env.sh"
CKPT=${1:?Provide this experiment checkpoint.pth}
set -- --headless
if [ "${HEADLESS:-1}" = 0 ]; then
  GUARD=$(python3 ../scripts/vulkan_gpu_guard.py "$RSI_PHYSICAL_GPU")
  export VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=$GUARD
  set --
fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_reference_rsi_time_late.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --num_agents 2 --num_envs "${EVAL_ENVS:-64}" --num_objects 4 \
  --output_path "${OUTPUT_PATH:-../runs/reference_rsi/time_late_eval}" \
  --task_graph "${TASK_GRAPH:-random_scenario}" --eval_skills loco --eval_skill_probs 1.0 \
  --eval_repeats 3 --test --eval --no_video "$@"
