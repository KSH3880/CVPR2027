#!/bin/bash
# Serve the plain Carry planner viewer through the shared loopback-only noVNC wrapper.
# Usage: PORT=6109 MA_GPU=0 view_vnc.sh <planner.pth> <frozen-ms18.pth>
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOCAL_VIEW="$ROOT/TokenHSI-coord/carry_planner/view.sh"
NOVNC_WRAPPER="$ROOT/scripts/masteer/view_sequential_stack.sh"

PLANNER=${1:?usage: view_vnc.sh <planner.pth> <frozen-ms18.pth>}
EXECUTOR=${2:?usage: view_vnc.sh <planner.pth> <frozen-ms18.pth>}

export NOVNC_LOCAL_VIEW="$LOCAL_VIEW"
exec bash "$NOVNC_WRAPPER" "$PLANNER" "$EXECUTOR"
