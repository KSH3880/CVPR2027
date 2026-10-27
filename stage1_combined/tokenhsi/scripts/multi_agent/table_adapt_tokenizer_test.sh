#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ "$#" -lt 1 ]; then echo "usage: $0 checkpoint.pth" >&2; exit 2; fi
CKPT=$(realpath -e "$1")
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
. "$SCRIPT_DIR/runtime_env.sh"
export OUTPUT_PATH=${OUTPUT_PATH:-$TOKENHSI_ROOT/../runs/table_adapt_tokenizer_eval_$(date +%Y%m%d_%H%M%S)/output_etc/check/eval}
mkdir -p "$OUTPUT_PATH"
python - "$CKPT" "$OUTPUT_PATH/env.yaml" "$OUTPUT_PATH/train.yaml" <<'CFG'
import sys,torch,yaml
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['relation_experiment_config']
cfg={k:c[k] for k in ('env','sim')};cfg['env']['holdAtFT']['asset_split']='test'
Path(sys.argv[2]).write_text(yaml.safe_dump(cfg,sort_keys=False))
train=c['train'];train['params']['algo']['name']='ma'
Path(sys.argv[3]).write_text(yaml.safe_dump(train,sort_keys=False))
CFG
if [ "${HEADLESS:-1}" = 0 ]; then . "$SCRIPT_DIR/gui_gpu_env.sh"; fi
set -- --task_graph holding_at --seed "${SEED:-42}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/run.py --task HumanoidTableTokenHSIFT \
 --cfg_train "$OUTPUT_PATH/train.yaml" --cfg_env "$OUTPUT_PATH/env.yaml" \
 --motion_file tokenhsi/data/dataset_table_hold_at.yaml --checkpoint "$CKPT" \
 --num_agents 1 --num_envs "${TABLE_TEST_ENVS:-16}" --num_objects 1 \
 --test --eval --eval_repeats "${TABLE_REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 \
 --output_path "$OUTPUT_PATH" --no_video "$@"
