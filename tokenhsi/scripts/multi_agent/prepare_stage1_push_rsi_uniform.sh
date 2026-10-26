#!/bin/bash
# CPU-only prerequisites + 78000-to-PUSH initialization. Never starts training.
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
. "$scripts/runtime_env.sh"
export CUDA_VISIBLE_DEVICES=""
port="$TOKENHSI_ROOT/ports/stage1_base_push"
export PYTHONPATH="$port/tokenhsi:$port"
source_checkpoint=${SOURCE_CHECKPOINT:-$port/weights/Stage1V2Fixed_00078000.pth}
cd "$port"
python check_rsi_uniform_data.py --source "$source_checkpoint"
[ "${CHECK_ONLY:-0}" = 0 ] || exit 0
python prepare_finetune.py transfer --variant stage1_push_rsi_uniform --source "$source_checkpoint" "$@"
