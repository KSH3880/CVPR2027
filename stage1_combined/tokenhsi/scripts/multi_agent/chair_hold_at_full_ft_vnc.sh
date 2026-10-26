#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-1}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PORT=${PORT:-6083} CHAIR_TEST_ENVS=1 CHAIR_REPEATS=${CHAIR_REPEATS:-20}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/chair_hold_at_full_ft_test.sh" "$1"
