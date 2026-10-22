#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
preflight
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1 HEADLESS=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/attn_view/native_y2m_4agent_boxes${JOINTPUSH_BOXES:-8}_$(date +%Y%m%d_%H%M%S)}
export PROBE_AGENTS=${PROBE_AGENTS:-4}
export JOINTPUSH_BOXES=${JOINTPUSH_BOXES:-8}
exec python "$JOINTPUSH_ROOT/scripts/attention_capture.py" --task HumanoidMAOpenPush \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_native_phase_y2m.yaml \
 --motion_file tokenhsi/data/dataset_push_native_phase_xy.yaml --checkpoint "$CHECKPOINT" \
 --num_agents "$PROBE_AGENTS" --num_objects "$JOINTPUSH_BOXES" --num_envs "${NUM_ENVS:-4}" \
 --task_graph push --eval_skills loco --eval_skill_probs 1.0 --test --eval --headless --no_video --eval_repeats 1 \
 --output_path "$OUTPUT_PATH" --seed "${SEED:-2141}" --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1
