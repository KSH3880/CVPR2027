#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="${TOKENHSI_RUNTIME:-/home/injesus1010/anaconda3/envs/tokenhsi118}"
export PATH="$runtime/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD:$PWD/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CXX=/usr/bin/g++-11 MAX_JOBS=2
export TOKENHSI_LOCAL_CUDA_COMPAT=1
export TOKENHSI_TRAIN_ENTRYPOINT=scripts/train_local.py
export PYTHONUNBUFFERED=1
exec bash scripts/train.sh
