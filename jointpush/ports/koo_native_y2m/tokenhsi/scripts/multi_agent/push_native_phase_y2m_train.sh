#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-1}
# [num_agents=2] [num_envs=2048] [num_objects=2]
if [ -z "${TOKENHSI_CONDA_ENV:-}" ]; then
  case "${CONDA_DEFAULT_ENV:-}" in
    ''|base) export TOKENHSI_CONDA_ENV=tokenhsi_koo ;;
    *) export TOKENHSI_CONDA_ENV=$CONDA_DEFAULT_ENV ;;
  esac
fi
export HEADLESS=1
unset TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX TOKENHSI_VIEWER_PUSH_LENGTH_Y TOKENHSI_VIEWER_LOW_VRAM TOKENHSI_EVAL_TRAIN_RSI
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth}
[ -f "$RESUME_CHECKPOINT" ] || { echo "Prepare the explicit Y2m resume checkpoint first: $RESUME_CHECKPOINT" >&2; exit 1; }
# This local 3090 cannot fit another 2048-env run alongside the source run.
# Read process metadata only; never stop source training or viewers here.
target_uuid=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader) || exit 1
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader) || exit 1
if printf '%s\n' "$gpu_processes" | awk -F, -v gpu="$target_uuid" '
  {gsub(/^[ \t]+|[ \t]+$/, "", $1)}
  $1==gpu && $3~/python/ {found=1}
  END {exit !found}'; then
  echo "GPU $TOKENHSI_GPU already has a Python CUDA client. Stop the intended source training/viewer yourself before Y2m resume; no process was stopped." >&2
  exit 1
fi
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-2}
set -- --task_graph push
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_y2m.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml \
  --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/push_native_phase_y2m}" \
  --experiment PushNativePhaseY2m --headless --no_video "$@"
