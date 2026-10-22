#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENHSI_VIEWER_QUIET=1
exec python "$JOINTPUSH_ROOT/scripts/native4_view.py" --task HumanoidMAOpenPush \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_y2m.yaml \
 --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml --checkpoint "$CHECKPOINT" \
 --num_agents 4 --num_objects "${JOINTPUSH_BOXES:-4}" --num_envs 1 --task_graph push --task_camera stack \
 --eval_skills loco --eval_skill_probs 1.0 --test --eval --no_video --eval_repeats "${REPEATS:-10000}" \
 --output_path "$OUTPUT_PATH" --seed "${SEED:-2141}"
