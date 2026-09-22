#!/bin/bash
# C2/simple MLP coordinator on the current Carry convergence stress layout.
# Usage: view_converge.sh <coordinator.pth> <frozen-ms18.pth> [envs]
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD_CKPT_INPUT=${1:?usage: view_converge.sh <coordinator.pth> <frozen-ms18.pth> [envs]}
EXEC_CKPT=${2:?usage: view_converge.sh <coordinator.pth> <frozen-ms18.pth> [envs]}
export ENVS=${ENVS:-${3:-1}}

for file in "$COORD_CKPT_INPUT" "$EXEC_CKPT"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 2; }
done
export COORD_CKPT=$(realpath -- "$COORD_CKPT_INPUT")
EXEC_CKPT=$(realpath -- "$EXEC_CKPT")

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

if [ -z "${COORD_MODEL:-}" ]; then
    COORD_MODEL=$(python - "$COORD_CKPT" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
kind = payload.get("model_kind", "")
schema = payload.get("schema_version", "")
if kind == "simple" or "simple" in schema:
    print("simple")
elif kind == "c2" or "c2" in schema:
    print("c2")
else:
    print("c1")
PY
    )
fi
export COORD_MODEL
if [ -z "${MS_CKPT:-}" ] && [ -f "$ROOT/TokenHSI-masteer/output/ckpt_stage1.pth" ]; then
    export MS_CKPT="$ROOT/TokenHSI-masteer/output/ckpt_stage1.pth"
fi
export MS_VIEW_CONVERGE=1
export MS_VIEW_CONVERGE_PROB=${MS_VIEW_CONVERGE_PROB:-1.0}
export MS_VIEW_CONVERGE_MARGIN=${MS_VIEW_CONVERGE_MARGIN:-0.25}
export MS_VIEW_TIMED_CROSS=0
export MS_SCEN=cross

echo "convergence viewer: model=$COORD_MODEL prob=$MS_VIEW_CONVERGE_PROB margin=${MS_VIEW_CONVERGE_MARGIN}m"
exec bash "$ROOT/scripts/coord/view_local.sh" "$EXEC_CKPT" "$ENVS"
