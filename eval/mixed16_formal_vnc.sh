#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)

for argument in "$@"; do
  case "$argument" in
    --help|-h|--dry-run) exec python3 "$ROOT/eval/mixed16_formal_viewer.py" "$@" ;;
  esac
done

export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in
  5|6|7) ;;
  *) echo 'Use TOKENHSI_GPU=5, 6 or 7.' >&2; exit 1 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
. "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index,memory.used,memory.total --format=csv
exec bash "$ROOT/tokenhsi/scripts/multi_agent/run-gui.sh" \
  python "$ROOT/eval/mixed16_formal_viewer.py" "$@"
