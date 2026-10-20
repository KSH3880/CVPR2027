#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
exec python tokenhsi/scripts/multi_agent/export_partition_reference_videos.py "$@"
