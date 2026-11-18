#!/bin/bash
# Limit CPU threads for warehouse training and evaluation.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
source "$(dirname "${BASH_SOURCE[0]}")/warehouse_waypoint_env.sh"
