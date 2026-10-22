#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
if [ "${HEADLESS:-0}" != 1 ]; then
  if [ -z "${DISPLAY:-}" ]; then
    echo "DISPLAY가 없습니다. 서버에서는 TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_reset_vnc.sh 를 사용하세요." >&2
    exit 1
  fi
  . "$(dirname "$0")/gui_gpu_env.sh"
fi
. "$(dirname "$0")/runtime_env.sh"
set -- --task HumanoidMADoor \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_unified_approach_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_unified_approach_amp.yaml \
  --motion_file tokenhsi/data/dataset_door_loco.yaml \
  --num_agents 2 --num_envs 1 --num_objects 2 --test --eval \
  --task_graph open --no_video --seed 42
if [ "${HEADLESS:-0}" = 1 ]; then set -- "$@" --headless; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/view_door_reset.py "$@"
