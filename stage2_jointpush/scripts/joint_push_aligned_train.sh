#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RSI=${RSI:-loco}
case "$RSI" in loco|koo) ;; *) echo "RSI must be loco or koo" >&2; exit 2 ;; esac
exec bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/joint_push_aligned_${RSI}_rsi_ca4_train.sh" "$@"
