#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
source "$(dirname "$0")/runtime_env.sh"
if [ "${HEADLESS:-0}" = 1 ]; then set -- "$@" --headless; fi
exec python tokenhsi/qualitative/bridge_rear90_zeroshot.py "$@"
