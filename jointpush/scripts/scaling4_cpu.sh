#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
export PROBE_AGENTS=4 TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export NUM_ENVS=${NUM_ENVS:-2} PROBE_STEPS=${PROBE_STEPS:-24}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
export SCALING_CPU_ARGS='--sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1'
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/scaling4_cpu_$(date +%Y%m%d_%H%M%S)}
exec bash "$JOINTPUSH_ROOT/scripts/scaling_eval.sh" "$CHECKPOINT"
