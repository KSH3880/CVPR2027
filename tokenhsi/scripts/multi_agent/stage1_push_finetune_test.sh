#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_STAGE1_PUSH_VARIANT=stage1_push_finetune
export OUTPUT_PATH=${OUTPUT_PATH:-output/stage1_base_push_finetune_stable_eval}
. "$scripts/stage1_base_push_common.sh"
stage1_base_push_run test "$@"
