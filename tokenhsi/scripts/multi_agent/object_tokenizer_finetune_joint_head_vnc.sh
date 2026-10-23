#!/bin/bash
set -eu
script_dir=$(cd "$(dirname "$0")" && pwd)
export HEADLESS=0
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
exec bash "$script_dir/run-gui.sh" bash "$script_dir/object_tokenizer_finetune_joint_head_test.sh" "$@"
