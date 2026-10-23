#!/bin/bash
# Evaluation only; no training config or checkpoint conversion.
export TOKENHSI_GPU=${TOKENHSI_GPU:-1}
export TOKENHSI_CONDA_ENV=tokenhsi
export PYTHONDONTWRITEBYTECODE=1
. "$(dirname "$0")/runtime_env.sh"
set -eu
set -- "$@"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
exec python tokenhsi/scripts/multi_agent/evaluate_object_zero_shot.py "$@"
