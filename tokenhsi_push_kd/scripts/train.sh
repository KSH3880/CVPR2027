#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly after checking existing GPU workloads}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/output_etc/check/torch_extensions}"
if [[ "${CHECK:-0}" == 1 ]]; then export MAX_ITERATIONS="${MAX_ITERATIONS:-1}"; fi
if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    export OUTPUT_PATH="${OUTPUT_PATH:-output_etc/check/smoke}"
    python - "$OUTPUT_PATH" <<'PY'
import sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
allowed = (Path.cwd() / 'output_etc/check').resolve()
if allowed not in target.parents:
    raise SystemExit('Short-run OUTPUT_PATH must be under output_etc/check/')
PY
fi
export CUDA_VISIBLE_DEVICES="$TOKENHSI_GPU"
export PYTHONPATH="$PWD:$PWD/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
args=(--task HumanoidTokenHSIPushKD --cfg_env tokenhsi/data/cfg/multi_task/push_kd.yaml --cfg_train tokenhsi/data/cfg/train/rlg/push_kd.yaml --motion_file tokenhsi/data/dataset_push_kd.yaml --num_envs 2048 --headless --sim_device cuda:0 --rl_device cuda:0 --pipeline gpu --seed "${SEED:-42}" --output_path "${OUTPUT_PATH:-output/push_kd}")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then args+=(--resume 1 --checkpoint "$RESUME_CHECKPOINT"); fi
if [[ -n "${MAX_ITERATIONS:-}" ]]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
exec python "${TOKENHSI_TRAIN_ENTRYPOINT:-tokenhsi/run.py}" "${args[@]}"
