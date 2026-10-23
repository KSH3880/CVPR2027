#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
if [ $# -lt 1 ] || [ ! -f "$1" ]; then echo 'usage: measure_push_hand_motion.sh checkpoint.pth [envs=64] [repeats=2]' >&2; exit 1; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/measure_push_hand_motion.py \
 --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env "${CFG_ENV:-tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal25.yaml}" \
 --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint "$1" \
 --num_agents 2 --num_envs "${2:-64}" --num_objects 4 --task_graph push \
 --test --eval --headless --no_video --seed "${SEED:-42}" --episode_length "${MOTION_EPISODE_LENGTH:-600}" --eval_skills loco --eval_skill_probs 1.0 --eval_repeats "${3:-2}" \
 --output_path "${OUTPUT_PATH:-output/push_hand_motion_9500}"
