#!/bin/bash
# Re-run one archived single-person evaluation initial state with its saved policy.
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
mode=${1:-direct}; seed=${2:-0}; scene=${3:-3}
case "$mode" in direct|around) ;; *) echo 'Use direct or around' >&2; exit 1 ;; esac
archive="$root/output_etc/check/push_starts_ft10000_20261026_run"
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_GUI_GPU_MODE=${TOKENHSI_GUI_GPU_MODE:-native}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo} HEADLESS=0
export TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_QUIET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1 PYTHONUNBUFFERED=1
export MPS_GPU_ROOT=/tmp/tokenhsi-standalone-evaluation
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
. "$scripts/gui_gpu_env.sh"
. "$scripts/runtime_env.sh"
export PYTHONPATH="$root/ports/stage1_base_push/tokenhsi:$root/ports/stage1_base_push"
output="$root/output_etc/check/push_eval_view_${mode}_seed${seed}_env${scene}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output"
cd "$root/ports/stage1_base_push"
exec python evaluate_single_match.py --benchmark-task push_at --push-start-mode "$mode" \
  --visualize --plan-env-index "$scene" --initial-plan "$archive/initial_states/${mode}_seed${seed}.npz" \
  --reference-checkpoint weights/Stage1V2Fixed_00078000.pth \
  --checkpoint "$root/output/stage1_base_push_finetune_stable_ep1200/Stage1PushFinetune_26-11-17-19/nn/Stage1PushFinetune_00010000.pth" \
  --task HumanoidMACarry --num_agents 1 --num_objects 1 --num_envs 1 --episode_length 1800 \
  --test --eval --no_video --seed "$seed" --task_camera stack \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --sim_device cuda:0 --pipeline gpu --rl_device cuda:0 --output_path "$output"
