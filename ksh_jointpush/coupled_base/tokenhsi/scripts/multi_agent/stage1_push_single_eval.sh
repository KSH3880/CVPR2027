#!/bin/bash
set -eu
CKPT=$1
shift
bash "$(dirname "$0")/evaluate.sh" stage1_push "$CKPT" --profile stage1_single "$@"
