#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=5 TOKENHSI_CONDA_ENV=tokenhsi_jhh PYTHONDONTWRITEBYTECODE=1
# CUDA 11.8 NVRTC cannot target RTX 5090; eager kernels work on this host.
export PYTORCH_JIT=${PYTORCH_JIT:-1}
export PORT=${PORT:-6083} CHAIR_TEST_ENVS=1 CHAIR_REPEATS=${CHAIR_REPEATS:-20}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/chair_sit_only_full_ft_test.sh" "$1"
