#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$scripts/runtime_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/tokenhsi-cpu-extensions}
port="$TOKENHSI_ROOT/ports/stage1_base_push"
export PYTHONPATH="$port/tokenhsi:$port"
cd "$port"
python audit_finetune_geometry.py
python check_finetune_rsi.py --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_finetune.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_finetune.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push_finetune.yaml \
  --num_agents 2 --num_envs 167 --num_objects 4 --headless --no_video --test \
  --task_graph random_scenario --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 8 \
  --output_path "$TOKENHSI_ROOT/output/stage1_base_push_finetune_stable_check"
