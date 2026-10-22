#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
WORK="$PROJECT_ROOT/runs/jointpush/five_task_500_fall_20261021/workspace"
MODEL=${MODEL:-depth180}
AGENTS=${PROBE_AGENTS:-2}
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU (-1 for CPU)}"
export HEADLESS=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
if [ "$TOKENHSI_GPU" = -1 ]; then
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
 export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
 CPU_ARGS='--sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1'
else
 CPU_ARGS=''
fi
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CKPT=$("$JOINTPUSH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["checkpoint"])' "$WORK/preparation.json" "$MODEL")
OUT=${OUTPUT_PATH:-$WORK/output_etc/check/${MODEL}_a${AGENTS}_$(date +%Y%m%d_%H%M%S)}
cd "$WORK"
exec python "$JOINTPUSH_ROOT/scripts/five_task_fall.py" --task HumanoidMAFiveTask \
 --cfg_env "$WORK/configs/${MODEL}_env.yaml" --cfg_train "$WORK/configs/${MODEL}_train.yaml" \
 --motion_file tokenhsi/data/dataset_five_task.yaml --checkpoint "$CKPT" --output_path "$OUT" \
 --num_agents "$AGENTS" --num_objects "$((2*AGENTS))" --num_envs "${NUM_ENVS:-64}" \
 --task_graph "${TASK_GRAPH:-push_at}" --eval_skills loco --eval_skill_probs 1.0 --eval_repeats 1 \
 --test --eval --headless --no_video --seed "${SEED:-2141}" $CPU_ARGS
