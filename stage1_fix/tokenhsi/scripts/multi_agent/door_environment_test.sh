#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
if [ "${HEADLESS:-1}" = 0 ]; then . "$(dirname "$0")/gui_gpu_env.sh"; fi
. "$(dirname "$0")/runtime_env.sh"
args=(--num-envs "${NUM_ENVS:-4}" --output "${OUTPUT_PATH:-output/door_environment_check}" --stiffness "${DOOR_STIFFNESS:-6.0}" --damping "${DOOR_DAMPING:-3.0}" --repeats "${REPEATS:-1}")
if [ "${HEADLESS:-1}" = 1 ]; then args+=(--headless); fi
python tokenhsi/door_environment.py "${args[@]}" "$@"
