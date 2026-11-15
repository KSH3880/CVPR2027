#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
. "$SCRIPT_DIR/output_paths.sh"
tokenhsi_select_output stage1_push_adaptive train
python tokenhsi/adaptive_train.py "$@"
