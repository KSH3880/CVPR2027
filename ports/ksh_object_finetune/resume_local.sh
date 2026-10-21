#!/usr/bin/env bash
# Run manually after GPU 0 is available. Never stops an existing training job.
set -euo pipefail
port_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$port_dir/../.." && pwd)
run_mode=${1:-check}
case "$run_mode" in
    check) final_epoch=501 ;;
    train) final_epoch=${MAX_ITERATIONS:-10000} ;;
    *) echo "Usage: bash ports/ksh_object_finetune/resume_local.sh [check|train]" >&2; exit 2 ;;
esac
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
checkpoint_path=${CHECKPOINT:-$port_dir/weights/resume_500.pth}
output_path=${OUTPUT_PATH:-$repo_dir/output/object_tokenizer_finetune/carry_chair_copied_adapter/resume_local_500_${run_mode}_$(date +%Y%m%d_%H%M%S)}
if [ ! -f "$checkpoint_path" ]; then
    echo "Checkpoint missing: $checkpoint_path" >&2; exit 1
fi
source "$repo_dir/tokenhsi/scripts/multi_agent/runtime_env.sh"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "Resume: $checkpoint_path -> $output_path (2048 envs, final epoch $final_epoch)"
python "$port_dir/launch.py" \
    --mode train --variant carry_chair_copied_adapter --envs 2048 \
    --checkpoint "$checkpoint_path" --output "$output_path" \
    --max-iterations "$final_epoch" --seed 42 --headless
