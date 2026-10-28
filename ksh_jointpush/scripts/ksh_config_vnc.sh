#!/usr/bin/env bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -n "${1:-}" ]; then
    RUN_DIR=$(python3 "$ROOT/ksh_jointpush/scripts/ksh_config_view_prepare.py" --checkpoint "$1")
else
    RUN_DIR=$(python3 "$ROOT/ksh_jointpush/scripts/ksh_config_view_prepare.py")
fi
echo "Prepared documented viewer: $RUN_DIR"
if [ "${PREPARE_ONLY:-0}" = 1 ]; then exit 0; fi
exec bash "$RUN_DIR/run.sh"
