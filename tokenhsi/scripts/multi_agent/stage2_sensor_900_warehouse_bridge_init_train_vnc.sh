#!/bin/bash
set -eu
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_warehouse_bridge_init train_vnc "$@"
