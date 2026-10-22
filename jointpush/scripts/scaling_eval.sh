#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
preflight
export HEADLESS=1 TOKENHSI_VIEWER_QUIET=1 PYTHONUNBUFFERED=1
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
AGENTS=${PROBE_AGENTS:-2}
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/scaling_${AGENTS}agent_$(date +%Y%m%d_%H%M%S)}
exec python "$JOINTPUSH_ROOT/scripts/scaling_eval.py" --task HumanoidMAOpenPush \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_y2m.yaml \
 --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml --checkpoint "$CHECKPOINT" \
 --num_agents "$AGENTS" --num_objects "${JOINTPUSH_BOXES:-$AGENTS}" --num_envs "${NUM_ENVS:-64}" \
 --task_graph push --eval_skills loco --eval_skill_probs 1.0 --test --eval --headless --no_video --eval_repeats 1 \
 --output_path "$OUT" --seed "${SEED:-2141}" ${SCALING_CPU_ARGS:-}
