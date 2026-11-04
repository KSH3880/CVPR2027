#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CKPT=$1
shift
bash "$SCRIPT_DIR/evaluate.sh" stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_radius3m "$CKPT" --profile general "$@"
