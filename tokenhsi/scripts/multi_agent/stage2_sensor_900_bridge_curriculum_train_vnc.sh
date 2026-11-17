#!/bin/bash
set -eu
bridge_gpu_index=$(nvidia-smi -i "${TOKENHSI_GPU:-0}" --query-gpu=index --format=csv,noheader,nounits)
source "$(dirname "$0")/../../../mps/shell.sh"
mps_use "$bridge_gpu_index"
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_bridge_curriculum train_vnc "$@"
