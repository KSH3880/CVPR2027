#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
MODE=${1:-train}
: "${KOO_PARTIAL_WORK:?Set KOO_PARTIAL_WORK to the prepared workspace}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
export HEADLESS=${HEADLESS:-1}
if [ "$MODE" = train ]; then
  gpu_uuid=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader)
  free=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  clients=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader)
  count=$(printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '{gsub(/^[ \t]+|[ \t]+$/,"",$1)} $1==gpu && $2~/python/ {n++} END{print n+0}')
  [ "$free" -ge 50000 ] && [ "$count" -le 1 ] || { echo 'Partial 2048-env launch requires >=50GiB free and at most one existing Python client.' >&2; exit 1; }
fi
if [ "$HEADLESS" = 0 ]; then export TOKENHSI_GUI_GPU_MODE=native; fi
. "$KOO_PORT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$KOO_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(--headless --sim_device cuda:0 --rl_device cuda:0)
if [ "${PARTIAL_CPU:-0}" = 1 ]; then
 export CUDA_VISIBLE_DEVICES=''
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
 args=(--headless --sim_device cpu --pipeline cpu --rl_device cpu)
elif [ "$MODE" = train ]; then
 [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] || { echo 'Verified GPU5 MPS attachment required' >&2; exit 1; }
fi
if [ "$HEADLESS" = 0 ]; then args=(--sim_device cuda:0 --rl_device cuda:0); fi
if [ "$MODE" = train ]; then
 NUM_ENVS=${NUM_ENVS:-2048}
 [ "$NUM_ENVS" = 2048 ] || { echo 'Keep training environments at 2048'; exit 1; }
 OUTPUT_PATH=${OUTPUT_PATH:-$KOO_PARTIAL_WORK/output/partial_training}
 if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--checkpoint "$RESUME_CHECKPOINT" --resume 1); fi
 if [ -n "${MAX_ITERATIONS:-}" ]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
else
 NUM_ENVS=${NUM_ENVS:-1}
 args+=(--test --eval --checkpoint "${CHECKPOINT:?Set CHECKPOINT for test}" --eval_skills loco --eval_skill_probs 1.0 --eval_repeats "${REPEATS:-100000}")
 OUTPUT_PATH=${OUTPUT_PATH:-$KOO_PARTIAL_WORK/output_etc/check/view}
fi
cd "$KOO_PARTIAL_WORK"
exec "$KOO_PYTHON" -u "$KOO_PROJECT/jointpush/scripts/koo_partial_run.py" --task HumanoidMAKooPartialPush \
 --cfg_env "$KOO_PARTIAL_WORK/configs/env.yaml" --cfg_train "$KOO_PARTIAL_WORK/configs/train.yaml" \
 --motion_file tokenhsi/data/dataset_stage1_base_push.yaml --num_agents 4 --num_objects 8 --num_envs "$NUM_ENVS" \
 --task_graph random_scenario --experiment KooPartialJointPush --seed "${SEED:-2141}" --no_video --output_path "$OUTPUT_PATH" "${args[@]}"
