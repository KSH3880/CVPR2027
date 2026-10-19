#!/usr/bin/env bash
source "$(dirname "$0")/runtime.sh"
set -- --task_graph joint_carry_at
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
 set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
elif [[ -n "${SOURCE_CHECKPOINT:-}" ]]; then
 set -- "$@" --transfer_checkpoint "$SOURCE_CHECKPOINT"
fi
if [[ -n "${MAX_ITERATIONS:-}" ]]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
exec python tokenhsi/run.py --task HumanoidMATableCarry \
 --cfg_env tokenhsi/data/cfg/multi_agent/table_undergrasp_carry.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/table_undergrasp_carry.yaml \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --num_agents 2 --num_envs 2048 --num_objects 4 \
 --sim_device cuda:0 --pipeline gpu --rl_device cuda:0 \
 --experiment TableUndergraspCarrySA4 --output_path "$OUTPUT_PATH" \
 --seed "${SEED:-42}" --headless --no_video "$@"
