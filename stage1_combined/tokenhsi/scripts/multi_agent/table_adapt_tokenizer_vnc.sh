#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PORT=${PORT:-6083} TABLE_TEST_ENVS=1 TABLE_REPEATS=${TABLE_REPEATS:-20}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/table_adapt_tokenizer_test.sh" "$1"
