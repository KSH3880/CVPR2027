#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi
export PORT=${PORT:-6086}
export VNC_DIR=/home/hwanhee/opt/vnc
export NOVNC_DIR=/home/hwanhee/opt/novnc
export WEBSOCKIFY=/home/hwanhee/anaconda3/envs/tokenhsi/bin/websockify
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ZERO_SHOT_POLICY_DEVICE=cuda:0
export NUM_ENVS=1 EVAL_REPEATS=10000 EVAL_SKILLS=${EVAL_SKILLS:-carryWith}
export OUTPUT_PATH=${OUTPUT_PATH:-/home/hwanhee/juan/CVPR2027/runs/jointcarry_zeroshot/view_6086}
exec bash "$ROOT/tokenhsi/scripts/multi_agent/run-gui.sh" bash "$ROOT/zeroshot/view_test.sh" "${1:-round_table}"
