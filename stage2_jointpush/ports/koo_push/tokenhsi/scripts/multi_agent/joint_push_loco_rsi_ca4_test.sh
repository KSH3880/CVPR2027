#!/bin/bash
set -eu
export TOKENHSI_GPU=6
. "$(dirname "$0")/runtime_env.sh"
PROJECT_ROOT=$(cd "$TOKENHSI_ROOT/../../.." && pwd)
CKPT=${1:?Provide a JointPush checkpoint}
# Restore the physics/reward/RSI contract used by this model, even when defaults changed.
EVAL_OUTPUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/evaluation}
set --
if [ "${JOINT_PUSH_REQUIRE_ALIGNED:-0}" = 1 ]; then set -- --require-aligned; fi
CFG_ENV=$(python tokenhsi/scripts/multi_agent/joint_push_checkpoint_env.py "$CKPT" \
 tokenhsi/data/cfg/multi_agent/joint_push_loco_rsi_ca4.yaml "$EVAL_OUTPUT/configs" "$@")
set -- --task_graph push
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
exec python tokenhsi/run.py --task HumanoidMAJointPush \
 --cfg_env "$CFG_ENV" \
 --cfg_train tokenhsi/data/cfg/train/rlg/joint_push_loco_rsi_ca4.yaml \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml \
 --checkpoint "$CKPT" --num_agents 2 --num_envs "${NUM_ENVS:-16}" --num_objects 4 \
 --eval_repeats "${REPEATS:-3}" --eval_skills loco --eval_skill_probs 1.0 \
 --output_path "$EVAL_OUTPUT" \
 --test --eval --no_video "$@"
