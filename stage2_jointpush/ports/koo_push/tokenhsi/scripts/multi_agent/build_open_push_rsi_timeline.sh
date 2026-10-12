#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
python tokenhsi/scripts/multi_agent/build_open_push_rsi_timeline.py "$@"
