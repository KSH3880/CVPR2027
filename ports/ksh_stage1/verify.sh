#!/bin/bash
set -eu
PORT_ROOT=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$PORT_ROOT/../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$REPO_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export TOKENHSI_KSH_PORT_TESTS=1
export PYTHONPATH="$PORT_ROOT/tokenhsi:$PORT_ROOT/tokenhsi/tests:$PORT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PORT_ROOT"
python -m pytest -q --import-mode=importlib \
  tokenhsi/tests/test_size_rsi.py \
  tokenhsi/tests/test_typed_edge_bias.py \
  tokenhsi/tests/test_typed_edge_message.py \
  tokenhsi/tests/test_stage1_unified.py \
  tokenhsi/tests/test_port_isolation.py
