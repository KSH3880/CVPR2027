#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROJECT="$ROOT/ksh_jointpush/coupled_base"
[ -f "$PROJECT/output_etc/check/branch_audit/preparation.json" ] || { echo 'Run coupled_base_prepare.py first' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$PROJECT/output_etc/check/torch_extensions"
exec bash "$PROJECT/tokenhsi/scripts/multi_agent/stage2_coupled_push_train.sh" "$@"
