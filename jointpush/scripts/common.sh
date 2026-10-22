#!/bin/bash
set -eu
JOINTPUSH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "$JOINTPUSH_ROOT/.." && pwd)
PORT_ROOT="$JOINTPUSH_ROOT/ports/koo_native_y2m"
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export PYTHONDONTWRITEBYTECODE=1
unset TOKENHSI_VIEWER_PUSH_LENGTH_Y TOKENHSI_EVAL_TRAIN_RSI
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/runs/jointpush/y2m/torch_extensions}
export CXX=${CXX:-/usr/bin/g++-11}
JOINTPUSH_PYTHON=${JOINTPUSH_PYTHON:-$HOME/anaconda3/envs/$TOKENHSI_CONDA_ENV/bin/python}
[ -x "$JOINTPUSH_PYTHON" ] || { echo "Set JOINTPUSH_PYTHON or TOKENHSI_CONDA_ENV for an installed Python." >&2; exit 1; }
resolve_checkpoint() {
    local input=${1:-${JOINTPUSH_CHECKPOINT:-$JOINTPUSH_ROOT/checkpoints/PushNativePhaseY2m_00006000.pth}}
    [ -f "$input" ] || { echo "Checkpoint missing: $input. Upload it to jointpush/checkpoints/ or pass its path." >&2; exit 1; }
    CHECKPOINT=$(realpath "$input")
}
preflight() {
    CUDA_VISIBLE_DEVICES='' "$JOINTPUSH_PYTHON" "$JOINTPUSH_ROOT/scripts/prepare.py" --checkpoint "$CHECKPOINT"
}
