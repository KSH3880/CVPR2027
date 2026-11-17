#!/bin/bash
# Evaluation-only Bridge BEFORE scene search; training code is untouched (runtime patches inside the script).
. "$(dirname "$0")/runtime_env.sh"
cd "$TOKENHSI_ROOT"
exec python tokenhsi/qualitative/bridge_before_search.py "$@"
