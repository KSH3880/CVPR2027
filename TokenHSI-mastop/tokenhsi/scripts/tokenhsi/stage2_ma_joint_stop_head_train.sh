#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

export MSTOP_INIT_MODE=stage1
export MSTOP_POLICY_MODE=joint_stop_head
export MSTOP_MIX=${MSTOP_MIX:-0.3,0.7,0,0}
export MSTOP_STOP_PROB=${MSTOP_STOP_PROB:-0.5714286}
export MSTOP_CARRYWITH=${MSTOP_CARRYWITH:-1}
export MSTOP_PHASE_REWARD=${MSTOP_PHASE_REWARD:-1}
export MSTOP_HOLD_STRICT_W=${MSTOP_HOLD_STRICT_W:-0.5}
export MSTOP_HOLD_CONTACT_W=${MSTOP_HOLD_CONTACT_W:-0.5}
export MSTOP_POST_DROP_C=${MSTOP_POST_DROP_C:-1.0}
export MSTOP_HOLD_SLIP_C=${MSTOP_HOLD_SLIP_C:-0.5}
export MSTOP_HOLD_LAT_C=${MSTOP_HOLD_LAT_C:-1.0}
export MSTOP_HOLD_YAW_C=${MSTOP_HOLD_YAW_C:-0.25}

exec bash "$HERE/stage2_ma_stop_residual_train.sh"
