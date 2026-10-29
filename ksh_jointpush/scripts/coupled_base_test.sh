#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROJECT="$ROOT/ksh_jointpush/coupled_base"
[ -f "$PROJECT/output_etc/check/branch_audit/preparation.json" ] || { echo 'Run coupled_base_prepare.py first' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$PROJECT/output_etc/check/torch_extensions"
# Graphics viewers use a separate CUDA context and never attach to MPS.
if [ "${HEADLESS:-0}" != 1 ] || [ "test" = vnc ]; then
  export MPS_GPU_ROOT="$PROJECT/output_etc/check/viewer_no_mps"
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
exec bash "$PROJECT/tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh" "$@"
