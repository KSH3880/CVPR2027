#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
port="$root/ports/stage1_base_push"
checkpoint=$(realpath -e "${1:?checkpoint required}")
output=$(realpath -m "${2:?new output directory required}")
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1 PYTHONUNBUFFERED=1
export HEADLESS=1 TOKENHSI_VIEWER_QUIET=1 MPS_GPU_ROOT=/tmp/tokenhsi-standalone-evaluation
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX TOKENHSI_VIEWER_LOW_VRAM
. "$scripts/runtime_env.sh"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
gpu_uuid=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader)
clients=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader)
if printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '
 {gsub(/^[ \t]+|[ \t]+$/, "", $1)} $1==gpu && $2~/python|mps-server/ {found=1} END {exit !found}'; then
  echo 'Evaluation GPU already has a CUDA client; left untouched.' >&2; exit 1
fi
free_mib=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free_mib" -ge 18000 ] || { echo 'Two concurrent evaluations require 18000 MiB free.' >&2; exit 1; }
mkdir -p "$output"
exec 9>"$output/eval.lock"
flock -n 9 || exit 1
export PYTHONPATH="$port/tokenhsi:$port"
cd "$port"
exec python evaluate_push_starts.py --checkpoint "$checkpoint" \
  --reference-checkpoint "$port/weights/Stage1V2Fixed_00078000.pth" --output "$output" \
  --num-envs "${TOKENHSI_EVAL_NUM_ENVS:-512}" --horizon "${TOKENHSI_EVAL_HORIZON:-1800}"
