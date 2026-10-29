#!/bin/bash
# Source after runtime_env.sh. CHECK=1 defaults to one short training check.
tokenhsi_select_output() {
  local experiment=$1 mode=$2
  if [ "${CHECK:-0}" = 1 ] && [ "$mode" = train ] && [ -z "${MAX_ITERATIONS:-}" ]; then
    MAX_ITERATIONS=1
  fi
  OUTPUT_PATH=$(python tokenhsi/utils/output_paths.py \
    --experiment "$experiment" --mode "$mode" --output "${OUTPUT_PATH:-}" \
    --check "${CHECK:-0}" --max-iterations "${MAX_ITERATIONS:-0}") || return
  export OUTPUT_PATH
}
