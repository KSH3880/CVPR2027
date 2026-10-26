#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_STAGE1_PUSH_VARIANT=stage1_push_finetune
export OUTPUT_PATH=${OUTPUT_PATH:-output/stage1_base_push_finetune_stable_eval}
export PORT=${PORT:-6108}
exec bash "$scripts/stage1_base_push_vnc.sh" "$@"
