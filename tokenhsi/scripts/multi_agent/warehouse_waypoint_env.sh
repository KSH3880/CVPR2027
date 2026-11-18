#!/bin/bash
# Sourced by the two inference wrappers; never starts or stops an MPS server.
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export TOKENHSI_GPU=${TOKENHSI_GPU:-7}
warehouse_gpu_index=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
case "$warehouse_gpu_index" in
  5|6|7) ;;
  *) echo 'Use TOKENHSI_GPU=5, 6 or 7.' >&2; exit 1 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/../../../mps/shell.sh"
case "${WAYPOINT_CUDA:-auto}" in
  auto)
    if mps_use "$warehouse_gpu_index"; then
      export WAYPOINT_CUDA=mps
    else
      export WAYPOINT_CUDA=direct
    fi ;;
  mps) mps_use "$warehouse_gpu_index" ;;
  direct) ;;
  *) echo 'WAYPOINT_CUDA must be auto, mps or direct.' >&2; exit 1 ;;
esac
if [ "$WAYPOINT_CUDA" = direct ]; then
    export TOKENHSI_GPU=$(nvidia-smi -i "$warehouse_gpu_index" --query-gpu=uuid --format=csv,noheader)
    export MPS_GPU_ROOT="/tmp/warehouse-waypoint-direct-${UID}-$$"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    echo "[waypoint] GPU $warehouse_gpu_index direct CUDA; existing MPS state left untouched"
fi
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
