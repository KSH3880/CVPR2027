#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-7}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
case "$TOKENHSI_GPU" in 5|6|7) ;; *) echo 'Use TOKENHSI_GPU=5, 6 or 7.' >&2; exit 1 ;; esac
source "$(dirname "$0")/../../../mps/shell.sh"
mps_use "$TOKENHSI_GPU"
export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/warehouse_close_random_heading_no_wall_cpa.yaml}
exec bash "$(dirname "$0")/launch.sh" warehouse_close_random_heading_no_wall_cpa train "$@"
