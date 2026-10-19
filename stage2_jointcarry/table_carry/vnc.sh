#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi
export PORT=${PORT:-6088} NUM_ENVS=1 HEADLESS=0 REPEATS=${REPEATS:-10000}
export VNC_DIR=${VNC_DIR:-/home/hwanhee/opt/vnc} NOVNC_DIR=${NOVNC_DIR:-/home/hwanhee/opt/novnc}
export WEBSOCKIFY=${WEBSOCKIFY:-/home/hwanhee/anaconda3/envs/tokenhsi/bin/websockify}
exec bash "$ROOT/tokenhsi/scripts/multi_agent/run-gui.sh" bash "$ROOT/table_carry/test.sh" "$@"
