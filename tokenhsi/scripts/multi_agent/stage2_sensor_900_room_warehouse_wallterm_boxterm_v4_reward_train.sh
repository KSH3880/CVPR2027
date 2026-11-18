#!/bin/bash
set -eu
export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward.yaml}
warehouse_gpu_index=$(nvidia-smi -i "${TOKENHSI_GPU:-0}" --query-gpu=index --format=csv,noheader,nounits)
source "$(dirname "$0")/../../../mps/shell.sh"
mps_use "$warehouse_gpu_index"
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward train "$@"
