#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CKPT=$1
shift
bash "$SCRIPT_DIR/evaluate.sh" stage2_before_nonemask_rsi433_near50 "$CKPT" --profile general "$@"
