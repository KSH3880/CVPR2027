#!/usr/bin/env bash
export TOKENHSI_GPU=5 TOKENHSI_CONDA_ENV=tokenhsi_jhh PYTHONDONTWRITEBYTECODE=1
# CUDA 11.8 NVRTC cannot target RTX 5090; eager kernels work on this host.
export PYTORCH_JIT=${PYTORCH_JIT:-1}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ "${HEADLESS:-1}" = 0 ]; then . "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/gui_gpu_env.sh"; fi
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output tokenhsi_sit_full_ft train
set -- --task_graph random_scenario --seed "${SEED:-42}"
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/run.py --task HumanoidTokenHSISitFT \
 --cfg_train tokenhsi/data/cfg/train/rlg/tokenhsi_sit_full_ft.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/tokenhsi_sit_full_ft.yaml \
 --motion_file tokenhsi/data/dataset_tokenhsi_sit_only.yaml \
 --num_agents 1 --num_envs 2048 --num_objects 1 \
 --output_path "$OUTPUT_PATH" --experiment TokenHSISitFullFT --no_video "$@"
