#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-0}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/table_hold_at_torch_extensions}
EVAL_PYTHON=${TOKENHSI_EVAL_PYTHON:-/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python}
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo "usage: $0 [checkpoint.pth] [eval options]"
    echo "No checkpoint: select the latest numbered Chair residual from active metadata."
    exec "$EVAL_PYTHON" "$ROOT/evaluation/hold_at/run_eval.py" --help
fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    CKPT=$(realpath -e "$1")
    shift
else
    CKPT=$("$EVAL_PYTHON" "$ROOT/evaluation/hold_at/cli_helpers.py" "$ROOT/chair_residual_active_run.json")
fi
echo "Checkpoint: $CKPT"
exec "$EVAL_PYTHON" "$ROOT/evaluation/hold_at/run_eval.py" \
 --checkpoint "$CKPT" --python "$EVAL_PYTHON" --gpu "${TOKENHSI_GPU:-0}" --concise "$@"
