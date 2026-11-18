#!/bin/bash
set -eu
export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward.yaml}
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward train_vnc "$@"
