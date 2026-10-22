#!/bin/bash
set -eu
KOO_PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
KOO_PORT="$KOO_PROJECT/jointpush/ports/koo_1021"
export KOO_PROJECT KOO_PORT
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$KOO_PROJECT/runs/jointpush/koo_1021/torch_extensions}
export CXX=${CXX:-/usr/bin/g++-11}
KOO_PYTHON=${KOO_PYTHON:-$HOME/anaconda3/envs/$TOKENHSI_CONDA_ENV/bin/python}
export PATH="$(dirname "$KOO_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$KOO_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
koo_checkpoint() {
 CHECKPOINT=${1:-${KOO_CHECKPOINT:-$KOO_PROJECT/jointpush/checkpoints/koo_1021/Stage1BasePush.pth}}
 [ -f "$CHECKPOINT" ] || { echo "Upload checkpoint to: $CHECKPOINT" >&2; exit 1; }
 CHECKPOINT=$(realpath "$CHECKPOINT")
 CUDA_VISIBLE_DEVICES='' "$KOO_PYTHON" "$KOO_PROJECT/jointpush/scripts/koo_1021_prepare.py" --checkpoint "$CHECKPOINT" --mode "${2:-test}"
}
