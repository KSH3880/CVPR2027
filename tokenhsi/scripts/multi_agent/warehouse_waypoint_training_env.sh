#!/bin/bash
# Select one of this machine's permitted GPUs, without touching existing clients.
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
waypoint_training_gpu=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
case "$waypoint_training_gpu" in 6) ;; *) echo 'Use TOKENHSI_GPU=6.' >&2; exit 1 ;; esac
source "$(dirname "${BASH_SOURCE[0]}")/warehouse_waypoint_env.sh"
