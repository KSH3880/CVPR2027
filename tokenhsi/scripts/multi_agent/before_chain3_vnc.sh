#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HEADLESS=0
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/before_chain3_test.sh" "$@"
