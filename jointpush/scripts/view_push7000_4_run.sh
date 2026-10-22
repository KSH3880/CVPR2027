#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$FIVE_TASK_VIEW_WORK"
exec python "$JOINTPUSH_ROOT/scripts/five_task_scaling_view.py" --task HumanoidMAFiveTask \
 --cfg_env "$FIVE_TASK_VIEW_WORK/configs/env.yaml" --cfg_train "$FIVE_TASK_VIEW_WORK/configs/train.yaml" \
 --motion_file tokenhsi/data/dataset_five_task.yaml --checkpoint "$FIVE_TASK_VIEW_CHECKPOINT" \
 --output_path "$OUTPUT_PATH" --num_agents 4 --num_objects 8 --num_envs 1 --task_graph push_at \
 --eval_skills loco --eval_skill_probs 1.0 --eval_repeats "${REPEATS:-10000}" \
 --test --eval --no_video --seed "${SEED:-2141}" \
 --sim_device "cuda:$TOKENHSI_GUI_GPU_INDEX" --rl_device "cuda:$TOKENHSI_GUI_GPU_INDEX"
