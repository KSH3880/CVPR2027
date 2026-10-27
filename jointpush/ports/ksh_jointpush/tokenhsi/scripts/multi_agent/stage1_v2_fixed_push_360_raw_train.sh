#!/bin/bash
set -eu
export STAGE1_PUSH_EXPERIMENT=stage1_v2_fixed_push_360_raw
export STAGE1_PUSH_RUN_NAME=Stage1V2FixedPush360Raw
exec bash "$(dirname "$0")/stage1_v2_fixed_push_train.sh" "$@"
