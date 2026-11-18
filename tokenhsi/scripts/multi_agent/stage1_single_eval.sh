#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CKPT=$1
shift
bash "$SCRIPT_DIR/evaluate.sh" stage1 "$CKPT" --profile stage1_single "$@"
