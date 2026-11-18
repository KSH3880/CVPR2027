#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export WAYPOINT_CUDA=${WAYPOINT_CUDA:-mps}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
waypoint_gpu_index=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
case "$waypoint_gpu_index" in
  6|7) ;;
  *) echo 'Use TOKENHSI_GPU=6 or 7.' >&2; exit 1 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/warehouse_waypoint_env.sh"
