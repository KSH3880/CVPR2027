#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi
source "$ROOT/tokenhsi/scripts/multi_agent/gui_gpu_env.sh"
export DISPLAY=:100
export ZERO_SHOT_POLICY_DEVICE=cuda:0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_ENVS=1 EVAL_REPEATS=10000 EVAL_SKILLS=carryWith
export OUTPUT_PATH=/home/hwanhee/juan/CVPR2027/runs/jointcarry_zeroshot/box_view_6086
exec bash "$ROOT/zeroshot/view_test.sh" box
