#!/bin/bash
set -eu
port=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$port/../.." && pwd)
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. "$root/tokenhsi/scripts/multi_agent/runtime_env.sh"
export TOKENHSI_STAGE1_BASE_PUSH_TESTS=1
export PYTHONPATH="$port/tokenhsi:$port/tokenhsi/tests:$port"
python -m pytest -q \
  "$port/tokenhsi/tests/test_stage1_push.py" \
  "$port/tokenhsi/tests/test_stage1.py" \
  "$port/tokenhsi/tests/test_stage1_policy.py" \
  "$port/tokenhsi/tests/test_stage1_gta.py" \
  "$port/tokenhsi/tests/test_stage1_v2_fixed.py" \
  "$port/tokenhsi/tests/test_stage1_v2.py" \
  "$port/tokenhsi/tests/test_stage1_sampling.py"
if [ "${WITH_SIM:-0}" = 1 ]; then
  bash "$root/tokenhsi/scripts/multi_agent/check_stage1_base_push.sh"
fi
