#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 2|5|6|7) ;; *) echo 'Use TOKENHSI_GPU=2, 5, 6 or 7.' >&2; exit 1 ;; esac
exec bash "$(dirname "$0")/evaluate.sh" stage2_distill_900_v5 "$@"
