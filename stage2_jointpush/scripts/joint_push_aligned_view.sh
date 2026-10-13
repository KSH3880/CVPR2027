#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export JOINT_PUSH_REQUIRE_ALIGNED=1
RSI=${RSI:-loco}
case "$RSI" in
 loco) SCRIPT=joint_push_view.sh ;;
 koo) SCRIPT=joint_push_rsi_view.sh ;;
 *) echo "RSI must be loco or koo" >&2; exit 2 ;;
esac
export PORT=${PORT:-6086}
exec bash "$ROOT/scripts/$SCRIPT" "$@"
