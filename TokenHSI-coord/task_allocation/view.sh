#!/bin/bash
# Usage: view.sh <0=local|1=VNC> <allocation.pth> [ms18.pth]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo 'usage: view.sh <0=local|1=VNC> <allocation.pth> [ms18.pth]'
    exit 0
fi
MODE=${1:?first argument: 0=local, 1=VNC}
shift
case "$MODE" in 0|1) ;; *) echo 'view mode must be 0 or 1' >&2; exit 2;; esac
[ $# -ge 1 ] && [ $# -le 2 ] || { echo 'expected allocation.pth [ms18.pth]' >&2; exit 2; }
[ -f "$1" ] || { echo "missing allocation checkpoint: $1" >&2; exit 2; }
# Internal log directory only; users need no experiment tag for viewing.
set -- "view_$(date +%Y%m%d_%H%M%S_%N)_$$" "$@"
export MA_GPU=${MA_GPU:-0}
export PORT=${PORT:-6109}
export ALLOC_VIEW=1
export ALLOC_ENVS=${ALLOC_ENVS:-1}
export ALLOC_MODE=eval
if [ "$MODE" = 0 ]; then
    export DISPLAY=${DISPLAY:-:0}
    exec bash "$ROOT/TokenHSI-coord/task_allocation/view_local.sh" "$@"
fi
export NOVNC_LOCAL_VIEW="$ROOT/TokenHSI-coord/task_allocation/view_local.sh"
exec bash "$ROOT/scripts/masteer/view_sequential_stack.sh" "$@"
