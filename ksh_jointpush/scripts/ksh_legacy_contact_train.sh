#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export KSH_CONTACT_REFERENCE_CONFIGS=${KSH_CONTACT_REFERENCE_CONFIGS:-$ROOT/runs/jointpush/ksh_jointpush_rsi/launch_base19500_hard_safe_clear_gpu5_20261028_214039/configs}
export KSH_CONTACT_MODE=legacy_jhh KSH_RSI_MODE=balanced KSH_GATE_MODE=hard
export KSH_INIT_SAFETY=all_pairs KSH_DISTRACTORS=clear_paths
exec bash "$(dirname "$0")/ksh_jointpush_launch.sh" train
