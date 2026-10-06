#!/bin/bash
# Evaluate 1050 with 0.3m component and union proximity metrics, outside MPS.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=${1:-0}
ENVS=${2:-64}
SEED=${3:-0}
STAGE_OVERRIDE=${MS_CKPT:-}
RUN="$ROOT/runs/carry_planner/carry_implicit_curvature05_lr3e4_clip20_3000_s0_r600_resume"
set -a
source "$RUN/run.env"
set +a
export MS_CKPT=${STAGE_OVERRIDE:-$stage1}
export MA_GPU="$GPU"
export CONDA_BASE=${CONDA_BASE:-/home/hwanhee/anaconda3}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export CARRY_PLANNER_CONVERGE_PROB=0.5
export CARRY_PLANNER_EVAL_PROXIMITY_THRESHOLD=0.3
export CARRY_PLANNER_EVAL_LABEL=proximity03
exec bash "$ROOT/TokenHSI-coord/carry_planner/eval_one.sh" \
 "$RUN/planner_001050.pth" \
 "$ROOT/TokenHSI-masteer/output/stack/ms18_maskteam_origscale_c06_s0_00009000.pth" \
 mixed "$ENVS" "$SEED"
