#!/bin/bash
set -eu
source "$(dirname "$0")/partial_common.sh"
MODE=${1:-train}
CUDA_VISIBLE_DEVICES='' "$PARTIAL_PYTHON" "$PARTIAL_ROOT/jointpush/scripts/partial_prepare.py" --verify
: "${PARTIAL_WORK:?Set PARTIAL_WORK to the prepared workspace}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
export HEADLESS=${HEADLESS:-1}
if [ "$MODE" = train ]; then
  gpu_uuid=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader)
  free=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  clients=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader)
  count=$(printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '{gsub(/^[ \t]+|[ \t]+$/,"",$1)} $1==gpu && $2~/python/ {n++} END{print n+0}')
  [ "$free" -ge 50000 ] && [ "$count" -le 1 ] || { echo 'Partial 2048-env launch requires >=50GiB free and at most one existing Python client.' >&2; exit 1; }
fi
if [ "$HEADLESS" = 0 ]; then
 export TOKENHSI_GUI_GPU_MODE=native
 . "$PARTIAL_ROOT/jointpush/ports/koo_native_y2m/tokenhsi/scripts/multi_agent/gui_gpu_env.sh"
 # Never attach a GUI process to a training MPS server.
 export MPS_GPU_ROOT=/tmp/partial-native-no-mps
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
if [ "${PARTIAL_CPU:-0}" = 1 ]; then
 export MPS_GPU_ROOT=/tmp/partial-cpu-no-mps
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
. "$PARTIAL_PORT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PARTIAL_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(--headless --sim_device cuda:0 --rl_device cuda:0)
if [ "${PARTIAL_CPU:-0}" = 1 ]; then
 export CUDA_VISIBLE_DEVICES=''
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
 args=(--headless --sim_device cpu --pipeline cpu --rl_device cpu)
elif [ "$MODE" = train ]; then
 [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] || { echo 'Verified selected-GPU MPS attachment required' >&2; exit 1; }
fi
if [ "$HEADLESS" = 0 ]; then
 # Match unfiltered NVIDIA Vulkan ordering: physical GPU index is the CUDA ordinal.
 export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
 args=(--sim_device "cuda:$TOKENHSI_GUI_GPU_INDEX" --rl_device "cuda:$TOKENHSI_GUI_GPU_INDEX")
 echo "[Partial native viewer] GPU=$TOKENHSI_GUI_GPU_INDEX CUDA/Vulkan/policy aligned; MPS bypassed"
fi
if [ "$MODE" = train ]; then
 NUM_ENVS=${NUM_ENVS:-2048}
 [ "$NUM_ENVS" = 2048 ] || { echo 'Keep training environments at 2048'; exit 1; }
 OUTPUT_PATH=${OUTPUT_PATH:-$PARTIAL_WORK/output/partial_training}
 if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--checkpoint "$RESUME_CHECKPOINT" --resume 1); fi
 if [ -n "${MAX_ITERATIONS:-}" ]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
else
 NUM_ENVS=${NUM_ENVS:-1}
 args+=(--test --eval --checkpoint "${CHECKPOINT:?Set CHECKPOINT for test}" --eval_skills loco --eval_skill_probs 1.0 --eval_repeats "${REPEATS:-100000}")
 OUTPUT_PATH=${OUTPUT_PATH:-$PARTIAL_WORK/output_etc/check/view}
fi
cd "$PARTIAL_WORK"
exec "$PARTIAL_PYTHON" -u "$PARTIAL_ROOT/jointpush/scripts/partial_run.py" --task PartialPushScene \
 --cfg_env "$PARTIAL_WORK/configs/env.yaml" --cfg_train "$PARTIAL_WORK/configs/train.yaml" \
 --motion_file tokenhsi/data/dataset_stage1_push.yaml --num_agents 4 --num_objects 8 --num_envs "$NUM_ENVS" \
 --task_graph random_scenario --experiment PartialRawCuboid --seed "${SEED:-2141}" --no_video --output_path "$OUTPUT_PATH" "${args[@]}"
