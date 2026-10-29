#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
python tokenhsi/evaluate.py "$@"
