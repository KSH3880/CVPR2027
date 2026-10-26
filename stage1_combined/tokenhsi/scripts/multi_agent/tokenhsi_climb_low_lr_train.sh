#!/usr/bin/env bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
# CUDA 11.8 NVRTC cannot target RTX 5090; eager kernels work on this host.
export PYTORCH_JIT=${PYTORCH_JIT:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ "${HEADLESS:-1}" = 0 ]; then . "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/gui_gpu_env.sh"; fi
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
if [ "${CHECK:-0}" = 1 ]; then MAX_ITERATIONS=5001; fi
tokenhsi_select_output tokenhsi_climb_low_lr train
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-../runs/tokenhsi_climb_78k_20261024/train/TokenHSIClimbFullFT_24-16-15-27/nn/TokenHSIClimbFullFT_00005000.pth}
if [ "${CHECK:-0}" = 1 ]; then MAX_ITERATIONS=5001; fi
set -- --task_graph random_scenario --seed "${SEED:-42}"
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/continue_climb.py --task HumanoidTokenHSIClimbFT \
 --cfg_train tokenhsi/data/cfg/train/rlg/tokenhsi_climb_low_lr.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/tokenhsi_climb_full_ft.yaml \
 --motion_file tokenhsi/data/dataset_amass_climb/dataset_amass_climb.yaml \
 --num_agents 1 --num_envs 2048 --num_objects 1 \
 --output_path "$OUTPUT_PATH" --experiment TokenHSIClimbLowLR --no_video "$@"
