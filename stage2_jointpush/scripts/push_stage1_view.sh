#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export TOKENHSI_GPU=6
export OUTPUT_PATH="$ROOT/runs/jointpush/push_stage1_repro_20261012/viewer"
export TASK_GRAPH=${TASK_GRAPH:-push}
export SEED=${SEED:-42}
export PORT=${PORT:-6081}
export WEBSOCKIFY=/home/hwanhee/anaconda3/envs/tokenhsi/bin/websockify
exec bash "$ROOT/stage2_jointpush/ports/koo_push/tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_vnc.sh" "$ROOT/runs/jointpush/push_stage1_repro_20261012/checkpoints/OpenPushPush180ContactAtWarmstart_00029000.pth" 2 "${NUM_ENVS:-1}" 4 "${REPEATS:-10000}"
