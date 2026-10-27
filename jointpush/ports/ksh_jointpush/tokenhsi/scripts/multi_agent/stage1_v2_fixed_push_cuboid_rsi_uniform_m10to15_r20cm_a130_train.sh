#!/bin/bash
set -eu
export STAGE1_PUSH_EXPERIMENT=stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
export STAGE1_PUSH_RUN_NAME=Stage1V2FixedPushCuboidRsiUniformM10to15R20cmA130
exec bash "$(dirname "$0")/stage1_v2_fixed_push_train.sh" "$@"
