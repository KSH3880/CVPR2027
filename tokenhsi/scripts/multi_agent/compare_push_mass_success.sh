#!/bin/bash
# Both policies evaluated concurrently on one GPU; does not start training.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TOKENHSI_GPU=${TOKENHSI_GPU:-4}
nvidia-smi -i "$TOKENHSI_GPU"
export TOKENHSI_CONDA_ENV=tokenhsi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source tokenhsi/scripts/multi_agent/runtime_env.sh
export PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
shopt -s nullglob
ck4=(output/open_push_contact_at_friction06_random_size_goal1to25_at01_resume6000_gpu4/*/nn/*_0???????.pth)
ck6=(output/push_only_physical_resume3500/*/nn/*_0???????.pth)
if (( ${#ck4[@]} == 0 || ${#ck6[@]} == 0 )); then
  echo 'Expected numbered checkpoints for both training runs.' >&2
  exit 1
fi
CHECKPOINT_GPU4=${CHECKPOINT_GPU4:-${ck4[${#ck4[@]}-1]}}
CHECKPOINT_GPU6=${CHECKPOINT_GPU6:-${ck6[${#ck6[@]}-1]}}
[[ -f "$CHECKPOINT_GPU4" && -f "$CHECKPOINT_GPU6" ]]
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  mkdir "$OUTPUT_PATH" # refuse overwriting existing results
else
  OUTPUT_PATH=$(mktemp -d output/push_mass_success_gpu${TOKENHSI_GPU}_XXXXXX)
fi
export OUTPUT_PATH
printf 'Results: %s\nGPU4 policy: %s\nGPU6 policy: %s\n' "$OUTPUT_PATH" "$CHECKPOINT_GPU4" "$CHECKPOINT_GPU6"
run_eval() {
  local label=$1 config=$2 motion=$3 checkpoint=$4
  python -u tokenhsi/scripts/multi_agent/evaluate_push_mass_success.py \
    --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
    --cfg_env "tokenhsi/data/cfg/multi_agent/$config.yaml" \
    --motion_file "tokenhsi/data/$motion.yaml" --checkpoint "$checkpoint" \
    --num_agents 2 --num_envs "${NUM_ENVS:-248}" --num_objects 4 --task_graph push \
    --test --eval --headless --no_video --seed "${SEED:-8429}" \
    --episode_length "${EPISODE_LENGTH:-600}" --eval_skills loco --eval_skill_probs 1.0 \
    --eval_repeats "${EVAL_REPEATS:-8}" --output_path "$OUTPUT_PATH/$label" \
    > "$OUTPUT_PATH/$label.log" 2>&1
}
run_eval gpu4_policy open_push_contact_at_friction06_random_size_goal1to25_at01 dataset_open_push_push180 "$CHECKPOINT_GPU4" &
pid4=$!
run_eval gpu6_policy open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors dataset_push_only_push180 "$CHECKPOINT_GPU6" &
pid6=$!
rc4=0; rc6=0
wait "$pid4" || rc4=$?
wait "$pid6" || rc6=$?
if (( rc4 != 0 || rc6 != 0 )); then
  printf 'Evaluation failed: gpu4_policy=%s gpu6_policy=%s; see %s/*.log\n' "$rc4" "$rc6" "$OUTPUT_PATH" >&2
  exit 1
fi
python tokenhsi/scripts/multi_agent/report_push_mass_success.py "$OUTPUT_PATH"
