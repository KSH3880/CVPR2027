#!/usr/bin/env bash
# Open a current stack-planner checkpoint in the native Isaac Gym viewer.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VIEWER="$ROOT/TokenHSI-coord/stack_planner/view.sh"
DEFAULT_POLICY="$ROOT/TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00011000.pth"

usage() {
    cat <<'EOF'
usage: ./visualize_planner.sh [options] [planner-checkpoint]

Options:
  --policy PATH       frozen executor checkpoint
  --envs N            number of scenes to display (default: 1)
  --gpu N             physical CUDA GPU (default: MA_GPU or 0)
  --graphics N        Vulkan graphics device (default: same as GPU)
  --check             validate compatibility without opening a window
  --list              list planner checkpoints and their schemas
  -h, --help          show this help

If planner-checkpoint is omitted, the newest checkpoint compatible with the
current stack_planner schema is selected.

Examples:
  ./visualize_planner.sh --list
  ./visualize_planner.sh --check runs/stack_planner/<tag>/planner_000010.pth
  ./visualize_planner.sh --gpu 1 runs/stack_planner/<tag>/planner_000010.pth
EOF
}

resolve_file() {
    local value=$1
    if [ -f "$value" ]; then
        realpath -- "$value"
    elif [ -f "$ROOT/$value" ]; then
        realpath -- "$ROOT/$value"
    else
        return 1
    fi
}

CONDA_BASE_VALUE=${CONDA_BASE:-}
if [ -z "$CONDA_BASE_VALUE" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE_VALUE=$(conda info --base)
    else
        CONDA_BASE_VALUE=/home/injesus1010/anaconda3
    fi
fi
ENV_NAME=${TOKENHSI_CONDA_ENV:-tokenhsi118}
PYTHON_BIN="$CONDA_BASE_VALUE/envs/$ENV_NAME/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    echo "Set CONDA_BASE and TOKENHSI_CONDA_ENV for this checkout." >&2
    exit 2
fi

list_checkpoints() {
    PYTHONPATH="$ROOT/TokenHSI-coord" "$PYTHON_BIN" - "$ROOT" <<'PY'
import sys
from pathlib import Path

import torch

from stack_planner.schema import STACK_SCHEMA_VERSION

root = Path(sys.argv[1])
paths = sorted(
    (root / "runs" / "stack_planner").glob("*/planner_*.pth"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(f"current schema: {STACK_SCHEMA_VERSION}")
print("compatible  schema                       step  checkpoint")
for path in paths:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        schema = payload.get("schema_version", "<missing>") if isinstance(payload, dict) else "<invalid>"
        step = payload.get("step", "-") if isinstance(payload, dict) else "-"
        compatible = "yes" if schema == STACK_SCHEMA_VERSION else "no"
        print(f"{compatible:10}  {schema:27}  {str(step):>4}  {path.relative_to(root)}")
    except Exception as error:
        print(f"error       {'<unreadable>':27}  {'-':>4}  {path.relative_to(root)} ({error})")
PY
}

POLICY=$DEFAULT_POLICY
ENVS=${ENVS:-1}
GPU=${MA_GPU:-0}
GRAPHICS=${TOKENHSI_GRAPHICS_DEVICE_ID:-}
CHECK_ONLY=0
LIST_ONLY=0
PLANNER=

while [ $# -gt 0 ]; do
    case "$1" in
        --policy)
            [ $# -ge 2 ] || { echo "--policy requires a path" >&2; exit 2; }
            POLICY=$2
            shift 2
            ;;
        --envs)
            [ $# -ge 2 ] || { echo "--envs requires a value" >&2; exit 2; }
            ENVS=$2
            shift 2
            ;;
        --gpu)
            [ $# -ge 2 ] || { echo "--gpu requires a value" >&2; exit 2; }
            GPU=$2
            shift 2
            ;;
        --graphics)
            [ $# -ge 2 ] || { echo "--graphics requires a value" >&2; exit 2; }
            GRAPHICS=$2
            shift 2
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            [ -z "$PLANNER" ] || { echo "only one planner checkpoint may be given" >&2; exit 2; }
            PLANNER=$1
            shift
            ;;
    esac
done

if [ $# -gt 0 ]; then
    [ -z "$PLANNER" ] && [ $# -eq 1 ] || { echo "unexpected arguments: $*" >&2; exit 2; }
    PLANNER=$1
fi

[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "--envs must be positive" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
if [ -z "$GRAPHICS" ]; then GRAPHICS=$GPU; fi
[[ "$GRAPHICS" =~ ^[0-9]+$ ]] || { echo "--graphics must be a non-negative integer" >&2; exit 2; }

if [ "$LIST_ONLY" -eq 1 ]; then
    list_checkpoints
    exit 0
fi

if [ -n "$PLANNER" ]; then
    original=$PLANNER
    PLANNER=$(resolve_file "$original") || { echo "planner checkpoint not found: $original" >&2; exit 2; }
else
    PLANNER=$(PYTHONPATH="$ROOT/TokenHSI-coord" "$PYTHON_BIN" - "$ROOT" <<'PY'
import sys
from pathlib import Path

import torch

from stack_planner.schema import STACK_SCHEMA_VERSION

root = Path(sys.argv[1])
for path in sorted((root / "runs" / "stack_planner").glob("*/planner_*.pth"),
                   key=lambda item: item.stat().st_mtime, reverse=True):
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("schema_version") == STACK_SCHEMA_VERSION:
            print(path.resolve())
            break
    except Exception:
        pass
PY
)
    if [ -z "$PLANNER" ]; then
        echo "No checkpoint matches the current stack planner schema." >&2
        echo "Run './visualize_planner.sh --list' to inspect available checkpoints." >&2
        exit 3
    fi
fi

original=$POLICY
POLICY=$(resolve_file "$original") || { echo "frozen policy not found: $original" >&2; exit 2; }

PYTHONPATH="$ROOT/TokenHSI-coord" "$PYTHON_BIN" - "$PLANNER" <<'PY'
import sys
from stack_planner.checkpoint import load_stack_checkpoint

try:
    model, payload = load_stack_checkpoint(sys.argv[1], "cpu")
except Exception as error:
    raise SystemExit(f"incompatible planner checkpoint: {error}")
print("planner checkpoint OK: schema={} step={} candidates={} history_steps={}".format(
    payload["schema_version"], payload.get("step", 0), model.config.candidates,
    model.config.history_steps,
))
PY

if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 0
fi

export CONDA_BASE=$CONDA_BASE_VALUE
export TOKENHSI_CONDA_ENV=$ENV_NAME
export MA_GPU=$GPU
export TOKENHSI_GRAPHICS_DEVICE_ID=$GRAPHICS

exec bash "$VIEWER" "$PLANNER" "$POLICY" "$ENVS"
