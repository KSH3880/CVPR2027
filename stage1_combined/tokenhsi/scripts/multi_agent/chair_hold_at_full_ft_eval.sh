#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
if [ "$#" -lt 1 ]; then echo 'usage: chair_hold_at_full_ft_eval.sh checkpoint [--prepare-only|--smoke] [--output runs/path]' >&2; exit 2; fi
CKPT=$1
shift
exec python3 "$ROOT/evaluation/hold_at/run_eval.py" --checkpoint "$CKPT" --gpu "${TOKENHSI_GPU:-5}" "$@"
