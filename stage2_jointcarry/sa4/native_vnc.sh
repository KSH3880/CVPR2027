#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi
export PORT=${PORT:-6088}
export VNC_DIR=${VNC_DIR:-/home/hwanhee/opt/vnc}
export NOVNC_DIR=${NOVNC_DIR:-/home/hwanhee/opt/novnc}
export WEBSOCKIFY=${WEBSOCKIFY:-/home/hwanhee/anaconda3/envs/tokenhsi/bin/websockify}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUM_ENVS=1 EVAL_REPEATS=${EVAL_REPEATS:-10000} HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointcarry_sa4/native_view_${PORT}_${2:-anchor}_${1:-teamhoi_default_rectangle}_$(date +%Y%m%d_%H%M%S)}
exec bash "$ROOT/tokenhsi/scripts/multi_agent/run-gui.sh" \
 bash "$ROOT/sa4/native_test.sh" "${1:-teamhoi_default_rectangle}" "${2:-anchor}"
