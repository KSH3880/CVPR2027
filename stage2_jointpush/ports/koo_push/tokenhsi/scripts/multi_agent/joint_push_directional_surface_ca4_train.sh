#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
if [ "${MAX_ITERATIONS:-}" = 0 ]; then echo "Use a positive MAX_ITERATIONS" >&2; exit 2; fi
. "$(dirname "$0")/runtime_env.sh"
PROJECT_ROOT=$(cd "$TOKENHSI_ROOT/../../.." && pwd)
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$PROJECT_ROOT/runs/jointpush/stage1_sources/OpenPushPush180ContactAtWarmstart_00029000.pth}
TRAIN_OUTPUT=${OUTPUT_PATH:-$PROJECT_ROOT/jointpush/output/directional_surface_training}
CFG_ENV=tokenhsi/data/cfg/multi_agent/joint_push_directional_surface_ca4.yaml
set -- --task_graph push
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  CFG_ENV=$(python tokenhsi/scripts/multi_agent/joint_push_checkpoint_env.py "$RESUME_CHECKPOINT" "$CFG_ENV" "$TRAIN_OUTPUT/configs" --require-random-shape)
  python - "$CFG_ENV" <<'CHECK'
import sys,yaml
s=yaml.safe_load(open(sys.argv[1]))['env']['jointPush']
if s.get('goal_mode')!='goal360' or s['reward'].get('contact_mode')!='directional_surface':
    raise ValueError('Resume requires the directional-surface profile')
CHECK
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
exec python tokenhsi/run.py --task HumanoidMAJointPushRandomShape \
 --cfg_env "$CFG_ENV" \
 --cfg_train tokenhsi/data/cfg/train/rlg/joint_push_directional_surface_ca4.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
 --num_agents 2 --num_envs 2048 --num_objects 4 \
 --output_path "$TRAIN_OUTPUT" \
 --experiment JointPushDirectionalSurfaceCa4 --headless --no_video "$@"
