#!/usr/bin/env bash
# Controlled physics probe; no training or policy/checkpoint changes.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)
cd "$repo_dir"
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
[[ "$TOKENHSI_GPU" == 6 ]] || { echo 'Use TOKENHSI_GPU=6.' >&2; exit 2; }
nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
output=$(realpath -m "${OUTPUT_PATH:-output_etc/check/push_depth180_high_$(date +%Y%m%d_%H%M%S)}")
case "$output" in "$repo_dir"/output_etc/check/*) ;; *) echo 'Output must be under output_etc/check/' >&2; exit 2 ;; esac
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
set +o pipefail
source "$script_dir/runtime_env.sh"
set -o pipefail
export PYTHONPATH="$repo_dir/tokenhsi:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
unset PUSH_RECT_QUICK PUSH_RECT_TRACK_REFERENCE
profile_mass=${PUSH_DEPTH_MASS:-20}
mkdir -p "$output"
python "$script_dir/analyze_push_rect_contacts.py" --depth180-sweep --profile-mass "$profile_mass" --absolute-heights .65 .95 1.0 1.1 1.2 --output "$output/contacts" > "$output/contacts.log" 2>&1
python "$script_dir/analyze_push_rect_contacts.py" --depth180-sweep --profile-mass "$profile_mass" --absolute-heights 1.12 1.15 --output "$output/contacts_boundary" > "$output/contacts_boundary.log" 2>&1
PUSH_RECT_DEPTH180=1 PUSH_RECT_DEPTH_MASS="$profile_mass" PUSH_RECT_INCLUDE_NATIVE=1 PUSH_RECT_ABSOLUTE_HEIGHTS=.65,.95,1.10,1.20 python "$script_dir/analyze_push_rect_humanoid.py" \
 --task HumanoidMAStage1Push --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_no_force.yaml --motion_file tokenhsi/data/dataset_stage1_push.yaml \
 --num_agents 2 --num_envs 294 --num_objects 4 --task_graph random_scenario --test --eval --headless --no_video \
 --output_path "$output/humanoid" > "$output/humanoid.log" 2>&1
python "$script_dir/report_push_depth180_high.py" "$output"
