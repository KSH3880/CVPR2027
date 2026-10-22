#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
export TOKENHSI_GPU=-1 CUDA_VISIBLE_DEVICES=-1 HEADLESS=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MPS_GPU_ROOT=/tmp/tokenhsi-cpu-no-mps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
source "$PORT_ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/attn_view/ksh_sitclimb2_boxes2_raw_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
for name in tokenhsi body_models lpanlib; do
 if [ ! -e "$OUT/$name" ]; then ln -s "/home/hwanhee/ksh/stage1_task/$name" "$OUT/$name"; fi
done
cd "$OUT"
exec python "$JOINTPUSH_ROOT/scripts/ksh_sitclimb_attention.py" --task HumanoidMACarry \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_distill.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint /home/hwanhee/ksh/stage1_task/output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_07-15-20-48/nn/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_00039000.pth \
 --num_agents "${PROBE_AGENTS:-2}" --num_objects "${JOINTPUSH_BOXES:-2}" --num_envs "${NUM_ENVS:-4}" --eval_repeats 1 \
 --eval_skills loco --eval_skill_probs 1.0 --task_graph random_scenario --output_path "$OUT" \
 --test --eval --no_video --headless --seed "${SEED:-42}" --sim_device cpu --pipeline cpu --rl_device cpu --num_threads 1
