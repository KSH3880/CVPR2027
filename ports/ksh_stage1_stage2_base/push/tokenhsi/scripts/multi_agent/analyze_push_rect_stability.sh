#!/usr/bin/env bash
# Analysis only: no policy training, config edits, or checkpoint writes.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)
cd "$repo_dir"
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
if [[ "$TOKENHSI_GPU" != 6 ]]; then
    echo 'This audited probe currently requires TOKENHSI_GPU=6.' >&2
    exit 2
fi
nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
output=$(realpath -m "${OUTPUT_PATH:-output_etc/check/push_rect_stability_$(date +%Y%m%d_%H%M%S)}")
case "$output" in "$repo_dir"/output_etc/check/*) ;; *) echo 'Output must be under output_etc/check/' >&2; exit 2 ;; esac
if [[ -e "$output" ]]; then echo "Output already exists: $output" >&2; exit 2; fi
source "$script_dir/runtime_env.sh"
export PYTHONPATH="$repo_dir/tokenhsi:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
unset PUSH_RECT_QUICK PUSH_RECT_TRACK_REFERENCE PUSH_RECT_ABSOLUTE_HEIGHTS
mkdir -p "$output"
python "$script_dir/analyze_push_rect_contacts.py" --output "$output/contacts" > "$output/contacts.log" 2>&1
python "$script_dir/analyze_push_rect_contacts.py" --height-fractions .55 .60 .65 --output "$output/contacts_high" > "$output/contacts_high.log" 2>&1
python "$script_dir/analyze_push_rect_contacts.py" --dense-mass 20 --absolute-height .65 --output "$output/contacts_dense_z065" > "$output/contacts_dense_z065.log" 2>&1
human_args=(--task HumanoidMAStage1Push --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml
    --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_no_force.yaml --motion_file tokenhsi/data/dataset_stage1_push.yaml
    --num_agents 2 --num_envs 2835 --num_objects 4 --task_graph random_scenario --test --eval --headless --no_video)
python "$script_dir/analyze_push_rect_humanoid.py" "${human_args[@]}" --output_path "$output/humanoid" > "$output/humanoid.log" 2>&1
PUSH_RECT_ABSOLUTE_HEIGHTS=.65 python "$script_dir/analyze_push_rect_humanoid.py" "${human_args[@]}" --output_path "$output/humanoid_z065" > "$output/humanoid_z065.log" 2>&1
python "$script_dir/report_push_rect_stability.py" "$output" > "$output/report.log" 2>&1
echo "$output/REPORT.md"
