#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
experiment_gpu_index=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits) || exit 1
case "$experiment_gpu_index" in 6|7) ;; *) echo 'Use TOKENHSI_GPU=6 or 7.' >&2; exit 1 ;; esac
exec bash "$(dirname "$0")/evaluate.sh" sct_coopft "$@"
