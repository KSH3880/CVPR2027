#!/bin/bash
set -eu
EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$EVAL_DIR/../.." && pwd)
EVAL_CALLER_DIR=${SINGLE_MATCH_CALLER_DIR:-$PWD}
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
source "$REPO_ROOT/ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/runtime_env.sh"
cd "$EVAL_CALLER_DIR"
exec python "$EVAL_DIR/evaluate.py" "$@"
