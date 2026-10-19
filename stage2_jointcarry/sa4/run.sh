#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODEL=${1:-anchor}
MODE=${2:-test}
case "$MODEL" in basic|anchor) ;; *) echo 'model: basic | anchor' >&2; exit 2;; esac
case "$MODE" in test|vnc) ;; *) echo 'mode: test | vnc' >&2; exit 2;; esac
if (( $# >= 2 )); then shift 2; else set --; fi
mapfile -t MODEL_PATHS < <(python3 - "$MODEL" "$MODE" <<'PY'
import json,sys
model=json.load(open('SA4_SOURCE.json'))['models'][sys.argv[1]]
print(model[sys.argv[2]+'_script'])
print(model['checkpoint'])
PY
)
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointcarry_sa4/${MODEL}_${MODE}_$(date +%Y%m%d_%H%M%S)}
exec bash "${MODEL_PATHS[0]}" "${MODEL_PATHS[1]}" "$@"
