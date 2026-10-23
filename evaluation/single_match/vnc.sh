#!/bin/bash
set -eu
EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$EVAL_DIR/../.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export PORT=${PORT:-6127}
export SINGLE_MATCH_CALLER_DIR=$PWD
exec bash "$REPO_ROOT/ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/run-gui.sh" \
    bash "$EVAL_DIR/run.sh" --viewer "$@"
