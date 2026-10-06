#!/bin/bash
# Usage: view.sh <0=local|1=VNC> <new-tag> <loco_carry|carryWith> [ms18.pth]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo 'usage: view.sh <0=local|1=VNC> <new-tag> <loco_carry|carryWith> [ms18.pth]'
    exit 0
fi
MODE=${1:?first argument: 0=local, 1=VNC}
shift
case "$MODE" in 0|1) ;; *) echo 'view mode must be 0 or 1' >&2; exit 2;; esac
[ $# -ge 2 ] && [ $# -le 3 ] || { echo 'expected new-tag skill [ms18.pth]' >&2; exit 2; }
case "$2" in loco_carry|carryWith) ;; *) echo 'skill must be loco_carry or carryWith' >&2; exit 2;; esac
export CARRY_BOX_SWAP_VIEW=1
export CARRY_BOX_SWAP_ENVS=${CARRY_BOX_SWAP_ENVS:-2}
if [ "$MODE" = 0 ]; then
    export DISPLAY=${DISPLAY:-:0}
    exec bash "$ROOT/TokenHSI-coord/carry_box_swap/launch_ms18.sh" "$@"
fi
export NOVNC_LOCAL_VIEW="$ROOT/TokenHSI-coord/carry_box_swap/view_local.sh"
exec bash "$ROOT/scripts/masteer/view_sequential_stack.sh" "$@"
