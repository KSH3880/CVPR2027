#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CKPT=$1
shift
bash "$SCRIPT_DIR/evaluate.sh" stage2_before_push_near50_pair50_source5 "$CKPT" --profile general "$@"
