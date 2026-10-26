#!/bin/bash
set -eu
. "$(dirname "$0")/runtime_env.sh"
exec python tokenhsi/scripts/multi_agent/push_uniform_18k_monitor.py "$@"
