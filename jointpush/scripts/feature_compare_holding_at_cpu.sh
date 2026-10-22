#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1 HEADLESS=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/attn_view/ksh_holding_at_feature_2vs4_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
for name in tokenhsi body_models lpanlib; do
 if [ ! -e "$OUT/$name" ]; then ln -s "/home/hwanhee/ksh/stage1_fix/$name" "$OUT/$name"; fi
done
cd "$OUT"
exec python "$JOINTPUSH_ROOT/scripts/feature_compare_holding_at.py" --task HumanoidMACarry \
 --cfg_train "$PROJECT_ROOT/runs/jointpush/ksh_unified6000_physics4/train.yaml" \
 --cfg_env "$PROJECT_ROOT/runs/jointpush/ksh_unified6000_physics4/env.yaml" \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint /home/hwanhee/ksh/stage1_fix/output/approach_scenario_stage1_unified/ApproachScenarioStage1Unified_30-15-57-04/nn/ApproachScenarioStage1Unified_00006000.pth \
 --num_agents 4 --num_objects 8 --num_envs "${NUM_ENVS:-32}" --eval_repeats 1 \
 --eval_skills loco --eval_skill_probs 1.0 --task_graph holding_at --output_path "$OUT" \
 --test --eval --no_video --headless --seed "${SEED:-2141}" --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1
