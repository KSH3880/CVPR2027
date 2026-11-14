#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 5|6) ;; *) echo 'Use TOKENHSI_GPU=5 or 6.' >&2; exit 1 ;; esac
exec bash "$(dirname "$0")/evaluate.sh" stage2_distill "$@"
