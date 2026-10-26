#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
port="$root/ports/stage1_base_push"
checkpoint=${1:-$root/output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth}
checkpoint=$(realpath -e "$checkpoint")
output=${2:-$root/output/stage1_base_push_finetune_eval9000_20261026}
output=$(realpath -m "$output")
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HEADLESS=1 TOKENHSI_VIEWER_QUIET=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX
unset TOKENHSI_VIEWER_PUSH_CUBES TOKENHSI_VIEWER_PUSH_CUBE_SIDE TOKENHSI_VIEWER_PUSH_CUBE_SIDES
unset TOKENHSI_VIEWER_PUSH_XY_RANDOM TOKENHSI_VIEWER_PUSH_BOX_RANGES TOKENHSI_VIEWER_PUSH_CUBE_FRICTION
# Dedicated native CUDA evaluation; do not join the running training's MPS.
export MPS_GPU_ROOT=/tmp/tokenhsi-standalone-evaluation
. "$scripts/runtime_env.sh"
gpu_uuid=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader)
clients=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)
eval_client_pids=$(printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '
  {gsub(/^[ \t]+|[ \t]+$/, "", $1)}
  $1==gpu && $3~/python/ {gsub(/[ \t]/,"",$2); print $2}')
if [ -n "$eval_client_pids" ]; then
  [ "${TOKENHSI_EVAL_CONCURRENT:-0}" = 1 ] || {
    echo 'Evaluation GPU already has a Python CUDA client; left untouched.' >&2; exit 1;
  }
  # Explicit opt-in shares only with this evaluator, never with training/unknown clients.
  for eval_client_pid in $eval_client_pids; do
    tr '\0' ' ' < "/proc/$eval_client_pid/cmdline" | grep -q 'evaluate_finetune.py' || {
      echo 'Concurrent GPU client is not this evaluator; left untouched.' >&2; exit 1;
    }
  done
  eval_free_mib=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$eval_free_mib" -ge 4096 ] || { echo 'Concurrent evaluation needs 4096 MiB free.' >&2; exit 1; }
fi
mkdir -p "$output"
exec 9>"$output/eval.lock"
flock -n 9 || { echo 'This evaluation is already running.' >&2; exit 1; }
export PYTHONPATH="$port/tokenhsi:$port"
cd "$port"
python preflight.py "$checkpoint" test stage1_push_finetune
eval_options=(--checkpoint "$checkpoint" --output "$output" --num-envs 512
              --push-episode-length "${TOKENHSI_EVAL_PUSH_EPISODE_LENGTH:-1200}")
if [ -n "${TOKENHSI_EVAL_TASKS:-}" ]; then
  read -r -a eval_tasks <<< "$TOKENHSI_EVAL_TASKS"
  eval_options+=(--tasks "${eval_tasks[@]}")
fi
exec python evaluate_finetune_batch.py "${eval_options[@]}"
