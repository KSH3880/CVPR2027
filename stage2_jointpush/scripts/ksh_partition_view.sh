#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_GUI_GPU_MODE=native
export TOKENHSI_CONDA_ENV=tokenhsi_juan
export PORT=${PORT:-6092} RESOLUTION=${RESOLUTION:-1600x900x24}
export VNC_DIR=${VNC_DIR:-/home/hwanhee/opt/vnc}
export WEBSOCKIFY=${WEBSOCKIFY:-/home/hwanhee/anaconda3/envs/tokenhsi_juan/bin/websockify}
prepare_args=(--workspace "${KSH_PARTITION_WORK:-$ROOT/runs/jointpush/ksh_partition_depth180_9000_20261022/workspace}")
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
  prepare_args+=(--checkpoint "$STAGE1_CHECKPOINT")
fi
python3 "$ROOT/stage2_jointpush/scripts/ksh_partition_prepare.py" "${prepare_args[@]}"
exec bash "$ROOT/jointpush/scripts/run_gui.sh" bash "$ROOT/stage2_jointpush/scripts/ksh_partition_view_run.sh"
