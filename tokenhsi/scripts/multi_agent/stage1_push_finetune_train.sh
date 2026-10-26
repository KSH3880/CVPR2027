#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
export TOKENHSI_STAGE1_PUSH_VARIANT=stage1_push_finetune
export TOKENHSI_TRAIN_EPISODE_LENGTH=${TOKENHSI_TRAIN_EPISODE_LENGTH:-1200}
export OUTPUT_PATH=${OUTPUT_PATH:-output/stage1_base_push_finetune_rsi_uniform}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$root/output/stage1_base_push_finetune_rsi_uniform/resume/Stage1PushFinetuneRsiUniform_from78000.pth}
[ -f "$RESUME_CHECKPOINT" ] || { echo 'First prepare and validate the 78000-to-PUSH transfer checkpoint.' >&2; exit 1; }
. "$scripts/stage1_base_push_common.sh"
stage1_base_push_run train "$@"
