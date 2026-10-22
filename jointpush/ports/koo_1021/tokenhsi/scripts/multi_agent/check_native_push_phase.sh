#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export TOKENHSI_GPU=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TORCH_EXTENSIONS_DIR=/tmp/tokenhsi-native-push-extensions
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=-1
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/check_native_push_phase.py \
 --task HumanoidMAOpenPush --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_xy.yaml --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml \
 --num_envs 36 --num_agents 2 --num_objects 2 --headless --no_video --test --eval \
 --task_graph push --num_threads 1 --sim_device cpu --pipeline cpu --rl_device cpu \
 --output_path "${OUTPUT_PATH:-output/push_native_phase_xy_forward_2to5m_check}"
