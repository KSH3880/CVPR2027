#!/bin/bash
set -eu
CKPT=$1
shift
bash "$(dirname "$0")/evaluate.sh" stage1_push_no_bias "$CKPT" --profile stage1_single "$@"
