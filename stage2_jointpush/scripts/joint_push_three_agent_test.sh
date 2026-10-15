#!/bin/bash
set -eu
export TOKENHSI_GPU=6
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT=${1:-$ROOT/../runs/jointpush/training_resume_20261012/JointPushLocoRsiCa4_12-19-45-11/nn/JointPushLocoRsiCa4_00009000.pth}
export JOINT_PUSH_THREE_AGENT_PREVIEW=1 NUM_AGENTS=3
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/three_agent_60kg_preview}
exec bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/joint_push_loco_rsi_ca4_test.sh" "$CKPT"
