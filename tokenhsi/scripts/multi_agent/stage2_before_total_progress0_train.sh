#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
SOURCE=${RESUME_CHECKPOINT:-output/stage2_before_total/Stage2BeforeTotal_06-22-22-14/nn/Stage2BeforeTotal_00009000.pth}
RESUME_CHECKPOINT=$(python tokenhsi/utils/before_total_progress_resume.py "$SOURCE")
export RESUME_CHECKPOINT
exec bash "$SCRIPT_DIR/launch.sh" stage2_before_total_progress0 train "$@"
