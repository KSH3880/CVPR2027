#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
. "$(dirname "$0")/runtime_env.sh"
set -eu
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ $# -lt 1 ] || [ ! -f "$1" ]; then echo 'Provide a Stage1 PUSH checkpoint.' >&2; exit 2; fi
CKPT=$1
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage1_push_face_xy_depth200_25kg eval
NUM_AGENTS=${2:-2}; NUM_ENVS=${3:-16}; NUM_OBJECTS=${4:-4}; NUM_REPEATS=${5:-3}
set -- --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" --eval_repeats "$NUM_REPEATS" --task_graph random_scenario --task_camera "${TASK_CAMERA:-stack}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
python tokenhsi/run.py --task HumanoidMAStage1Push \
 --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_face_xy_depth200_25kg.yaml --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_face_xy_depth200_25kg.yaml \
 --motion_file tokenhsi/data/dataset_stage1_push.yaml --checkpoint "$CKPT" --output_path "$OUTPUT_PATH" \
 --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" --seed "${SEED:-42}" --test --eval --no_video "$@"
