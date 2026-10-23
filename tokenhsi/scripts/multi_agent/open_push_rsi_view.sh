#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
RSI_GPU_INDEX=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits | tr -d '[:space:]')
if [ "$RSI_GPU_INDEX" != 6 ]; then echo "OPEN/PUSH uses GPU 6 only" >&2; exit 1; fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
if [ "${HEADLESS:-0}" != 1 ]; then
  if [ -z "${DISPLAY:-}" ]; then echo "서버에서는 open_push_rsi_view_vnc.sh를 사용하세요." >&2; exit 1; fi
  . "$(dirname "$0")/gui_gpu_env.sh"
fi
. "$(dirname "$0")/runtime_env.sh"
set -- --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env "${RSI_CFG_ENV:-tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml}" --motion_file "${RSI_MOTION_FILE:-tokenhsi/data/dataset_open_push.yaml}" \
 --num_agents 2 --num_envs 1 --num_objects 4 --test --eval --task_graph "${TASK_GRAPH:-open_push}" \
 --no_video --seed "${SEED:-42}" --output_path "${OUTPUT_PATH:-output/open_push_rsi_view}"
if [ "${HEADLESS:-0}" = 1 ]; then set -- "$@" --headless; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/view_open_push_rsi.py "$@"
