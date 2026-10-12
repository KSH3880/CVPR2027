#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/transfer_open_push_random_lanes.py "$@"
