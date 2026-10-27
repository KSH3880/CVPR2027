#!/bin/bash
set -eu
export STAGE1_PUSH_EXPERIMENT=stage1_v2_fixed_push_360_raw_mass20to25
exec bash "$(dirname "$0")/stage1_v2_fixed_push_test.sh" "$@"
