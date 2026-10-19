#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
if [ $# -lt 1 ] || [ ! -f "$1" ]; then echo 'usage: lift_stage1_teamhoi_test.sh CHECKPOINT [envs=16] [repeats=3]' >&2; exit 1; fi
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
lift_gpu_index=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
[ "$lift_gpu_index" = 6 ] || { echo 'Lift uses physical GPU 6 only' >&2; exit 1; }
if [ "${HEADLESS:-1}" = 0 ]; then . "$SCRIPT_DIR/gui_gpu_env.sh"; fi
. "$SCRIPT_DIR/runtime_env.sh"
args=()
if [ "${HEADLESS:-1}" != 0 ]; then args+=(--headless); fi
python tokenhsi/run.py --task HumanoidMALift \
 --cfg_env "${CFG_ENV:-tokenhsi/data/cfg/multi_agent/lift_stage1_teamhoi.yaml}" \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1_teamhoi.yaml \
 --motion_file tokenhsi/data/dataset_lift_stage1.yaml \
 --checkpoint "$1" --num_agents 2 --num_objects 2 --num_envs "${2:-16}" \
 --eval_repeats "${3:-3}" --test --eval --no_video \
 --output_path "${OUTPUT_PATH:-output/lift_stage1_teamhoi_eval}" "${args[@]}"
