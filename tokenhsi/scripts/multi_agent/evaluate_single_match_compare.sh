#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
port="$root/ports/stage1_base_push"
reference="$port/weights/Stage1V2Fixed_00078000.pth"
finetune="$root/output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth"
output=${1:-$root/output_etc/check/single_match_78000_vs_push9000_20261026}
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1 PYTHONUNBUFFERED=1
export HEADLESS=1 TOKENHSI_VIEWER_QUIET=1 MPS_GPU_ROOT=/tmp/tokenhsi-standalone-evaluation
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX TOKENHSI_VIEWER_LOW_VRAM
. "$scripts/runtime_env.sh"
gpu_uuid=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader)
clients=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader)
if printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '
 {gsub(/^[ \t]+|[ \t]+$/, "", $1)} $1==gpu && $2~/python/ {found=1} END {exit !found}'; then
  echo 'GPU has a Python CUDA client; left untouched.' >&2; exit 1
fi
mkdir -p "$output"
exec 9>"$output/eval.lock"
flock -n 9 || { echo 'Single-person comparison already running.' >&2; exit 1; }
export PYTHONPATH="$port/tokenhsi:$port"
cd "$port"
exec python evaluate_single_batch.py --reference-checkpoint "$reference" \
  --finetune-checkpoint "$finetune" --output "$output" --num-envs 512 --push-horizon 1800
