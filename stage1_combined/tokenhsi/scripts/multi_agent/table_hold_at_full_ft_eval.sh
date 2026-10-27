#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${TOKENHSI_EVAL_PYTHON:-/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python}
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    CKPT=$1
    shift
else
    CKPT=$("$PYTHON" - "$ROOT" <<'LATEST'
import json,sys,re
from pathlib import Path
root=Path(sys.argv[1])
active=json.loads((root/'table_hold_at_active_run.json').read_text())
folder=Path(active['experiment_dir'])/'nn'
paths=[p for p in folder.glob('TableHoldAtFullFT_*.pth') if re.fullmatch(r'TableHoldAtFullFT_\d+\.pth',p.name)]
if not paths: raise SystemExit('No numbered Table checkpoint saved yet; pass an explicit checkpoint.')
print(max(paths,key=lambda p:int(p.stem.rsplit('_',1)[1])))
LATEST
    )
fi
exec "$PYTHON" "$ROOT/evaluation/table_hold_at/run_eval.py" --checkpoint "$CKPT" --python "$PYTHON" --gpu "${TOKENHSI_GPU:-0}" "$@"
