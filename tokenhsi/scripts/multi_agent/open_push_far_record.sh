#!/bin/bash
# Clean camera recording on the same CUDA/Vulkan GPU, with a noVNC viewer.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export PORT=${PORT:-6106}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
if [ "${_FAR_RECORD_CHILD:-0}" != 1 ]; then
  exec bash "$SCRIPT_DIR/run-gui.sh" env _FAR_RECORD_CHILD=1 bash "$0" "$@"
fi
. "$SCRIPT_DIR/runtime_env.sh"
CKPT=${1:-output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00029000.pth}
if [ $# -gt 1 ] || [ ! -f "$CKPT" ]; then
  echo "usage: $0 [warmstart_checkpoint.pth]" >&2
  exit 1
fi
export CONTINUOUS_VIDEO=1
export PYTHONPATH="$ROOT/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$SCRIPT_DIR/view_open_push_far.py" \
  --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml \
  --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint "$CKPT" \
  --num_agents 2 --num_envs 1 --num_objects 4 \
  --task_graph push --task_camera agent \
  --eval_skills loco --eval_skill_probs 1.0 \
  --seed "${SEED:--1}" \
  --eval_repeats "${EVAL_REPEATS:-1000000000}" \
  --episode_length "${EPISODE_LENGTH:-1200}" \
  --output_path "${OUTPUT_PATH:-output_records}" \
  --test --eval
