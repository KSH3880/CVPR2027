#!/bin/bash
set -eu
# CPU-only: hide CUDA and bypass MPS discovery before sourcing runtime setup.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
extra=(--sim_device cpu --pipeline cpu --rl_device cpu)
export CUDA_VISIBLE_DEVICES=""
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/tokenhsi-cpu-extensions}
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_push_random_scene.py \
 --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_y2m.yaml --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml \
 --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video --test --eval \
 --task_graph push --num_threads 1 --output_path "${OUTPUT_PATH:-output/push_native_phase_y2m_check}" "${extra[@]}"
