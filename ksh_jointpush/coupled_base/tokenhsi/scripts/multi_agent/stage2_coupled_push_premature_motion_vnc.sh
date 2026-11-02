#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export TOKENHSI_RUN_ENTRY=${TOKENHSI_RUN_ENTRY:-$PROJECT/../scripts/coupled_base_run.py}
export TORCH_EXTENSIONS_DIR="$PROJECT/output_etc/check/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
export HEADLESS=0
if [ "${HEADLESS:-0}" != 1 ]; then
  export MPS_GPU_ROOT="$PROJECT/output_etc/check/viewer_no_mps"
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  . "$SCRIPT_DIR/gui_gpu_env.sh"
fi
exec bash "$SCRIPT_DIR/launch.sh" stage2_coupled_push_region_goals vnc "$@"
