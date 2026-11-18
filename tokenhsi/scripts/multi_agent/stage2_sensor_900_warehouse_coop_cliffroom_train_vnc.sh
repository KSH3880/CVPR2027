#!/bin/bash
set -eu
export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_warehouse_coop_cliffroom.yaml}
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_warehouse_coop_cliffroom train_vnc "$@"
