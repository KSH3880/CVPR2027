#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
. "$SCRIPT_DIR/gui_gpu_env.sh"
[ "$TOKENHSI_GUI_GPU_INDEX" = 6 ] || { echo 'Lift preview requires physical GPU 6' >&2; exit 1; }
. "$SCRIPT_DIR/runtime_env.sh"
exec python tokenhsi/scripts/multi_agent/lift_stage1_preview.py --task HumanoidMALift \
 --cfg_env tokenhsi/data/cfg/multi_agent/lift_stage1.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1.yaml \
 --motion_file tokenhsi/data/dataset_lift_stage1.yaml \
 --num_envs 1 --num_agents 2 --num_objects 2 --test --no_video \
 --output_path "${OUTPUT_PATH:-output/lift_stage1_preview}"
