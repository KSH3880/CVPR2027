#!/usr/bin/env bash
source "$(dirname "$0")/runtime.sh"
CKPT=${1:?Provide a table-carry checkpoint}
set -- --task_graph joint_carry_at
if [[ "${HEADLESS:-1}" != 0 ]]; then set -- "$@" --headless; fi
if [[ -n "${EPISODE_LENGTH:-}" ]]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
exec python tokenhsi/run.py --task HumanoidMATableCarry \
 --cfg_env tokenhsi/data/cfg/multi_agent/table_undergrasp_carry.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/table_undergrasp_carry.yaml \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
 --num_agents 2 --num_envs "${NUM_ENVS:-16}" --num_objects 4 \
 --eval_repeats "${REPEATS:-1}" --eval_skills carryWith --eval_skill_probs 1.0 \
 --sim_device cuda:0 --pipeline gpu --rl_device cuda:0 \
 --output_path "$OUTPUT_PATH" --seed "${SEED:-42}" --test --eval --no_video "$@"
