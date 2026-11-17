#!/bin/bash
set -eu
if [ -z "${RESUME_CHECKPOINT:-}" ]; then
  export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache.yaml}
fi
bridge_gpu_index=$(nvidia-smi -i "${TOKENHSI_GPU:-0}" --query-gpu=index --format=csv,noheader,nounits)
source "$(dirname "$0")/../../../mps/shell.sh"
mps_use "$bridge_gpu_index"
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_bridge_before_fixed_layout_cache train "$@"
