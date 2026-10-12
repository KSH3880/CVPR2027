#!/bin/bash
# Evaluation only: <checkpoint> [additional evaluate_z_sweep.py arguments]
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
exec python tokenhsi/scripts/multi_agent/evaluate_z_sweep.py "$@"
