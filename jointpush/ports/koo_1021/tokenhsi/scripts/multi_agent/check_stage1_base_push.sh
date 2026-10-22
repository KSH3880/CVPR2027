#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$scripts/runtime_env.sh"
root=$TOKENHSI_ROOT
port="$root/ports/stage1_base_push"
export CUDA_VISIBLE_DEVICES=""
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/tokenhsi-cpu-extensions}
export PYTHONPATH="$port/tokenhsi:$port"
cd "$port"
python check_scene.py --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_base_push.yaml \
  --motion_file tokenhsi/data/dataset_stage1_base_push.yaml \
  --num_agents 2 --num_envs 2 --num_objects 4 --headless --no_video --test \
  --task_graph random_scenario --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1 \
  --output_path "$root/output/stage1_base_push_check"
python check_player.py --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_base_push.yaml \
  --motion_file tokenhsi/data/dataset_stage1_base_push.yaml \
  --num_agents 2 --num_envs 2 --num_objects 4 --headless --no_video --test --eval \
  --task_graph push_at --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1 \
  --output_path "$root/output/stage1_base_push_check"
