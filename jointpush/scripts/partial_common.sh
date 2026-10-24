#!/bin/bash
set -eu
PARTIAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PARTIAL_PORT="$PARTIAL_ROOT/jointpush/ports/jhh_raw_cuboid"
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan} PYTHONDONTWRITEBYTECODE=1
PARTIAL_PYTHON=${PARTIAL_PYTHON:-$HOME/anaconda3/envs/$TOKENHSI_CONDA_ENV/bin/python}
export PATH="$(dirname "$PARTIAL_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PARTIAL_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PARTIAL_ROOT/runs/jointpush/torch_extensions}
export CXX=${CXX:-/usr/bin/g++-11}
