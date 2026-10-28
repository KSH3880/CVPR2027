#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
if [ "$#" -lt 1 ]; then echo "usage: $0 checkpoint.pth [eval options]" >&2; exit 2; fi
CKPT=$(realpath -e "$1")
shift
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
EVAL_PYTHON=${TOKENHSI_EVAL_PYTHON:-/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python}
exec "$EVAL_PYTHON" "$ROOT/evaluation/hold_at/run_eval.py" \
 --checkpoint "$CKPT" --python "$EVAL_PYTHON" --gpu "${TOKENHSI_GPU:-0}" "$@"
