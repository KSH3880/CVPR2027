#!/usr/bin/env bash
set -euo pipefail
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTORCH_JIT=0 PYTHONUNBUFFERED=1
export HEADLESS=1 SEED=0 CHAIR_TEST_ENVS=512 CHAIR_REPEATS=3
unset EPISODE_LENGTH TASK_GRAPH CHECK MAX_ITERATIONS RESUME_CHECKPOINT
OUT=${1:-output_etc/check/tokenhsi_climb_ft_sweep_20261025}
CKPT_DIR=../runs/tokenhsi_climb_78k_20261024/train/TokenHSIClimbFullFT_24-16-15-27/nn
for epoch in 05000 10000 15000; do
  output="$OUT/ft_$epoch"
  if [ -e "$output" ]; then echo "Existing output: $output" >&2; exit 2; fi
  mkdir -p "$output"
  OUTPUT_PATH="$output" bash tokenhsi/scripts/multi_agent/tokenhsi_climb_full_ft_test.sh "$CKPT_DIR/TokenHSIClimbFullFT_000$epoch.pth" > "$output/eval.log" 2>&1
  echo "FT $epoch rc=0"
done
