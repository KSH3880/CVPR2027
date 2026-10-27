#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
export PYTORCH_JIT=${PYTORCH_JIT:-0}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ "$#" -lt 1 ]; then echo "usage: $0 checkpoint.pth" >&2; exit 2; fi
CKPT=$(realpath -e "$1")
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
. "$SCRIPT_DIR/runtime_env.sh"
export OUTPUT_PATH=${OUTPUT_PATH:-$TOKENHSI_ROOT/../runs/table_carry_rsi_eval_$(date +%Y%m%d_%H%M%S)/output_etc/check/eval}
mkdir -p "$OUTPUT_PATH"
python - "$CKPT" "$OUTPUT_PATH/env.yaml" <<'CFG'
import sys,os,torch,yaml
from pathlib import Path
weights=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
c=weights['relation_experiment_config']
if 'tableCarryRsi' not in c['env']:
    if weights['epoch'] != 78000: raise ValueError('Use a Table RSI checkpoint or original 78k')
    cfg=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/table_carry_rsi_full_ft.yaml').read_text())
else: cfg={k:c[k] for k in ('env','sim')}
cfg['env']['holdAtFT']['asset_split']='test'
if os.environ.get('TABLE_RSI_VIEW')=='1':
    cfg['env']['tableCarryRsi']['probability']=1.
    bank=Path(os.environ.get('TABLE_RSI_BANK',cfg['env']['tableCarryRsi']['bank']))
    cfg['env']['holdAtFT']['object_root']=str(bank/'viewer_assets')
Path(sys.argv[2]).write_text(yaml.safe_dump(cfg,sort_keys=False))
CFG
if [ "${HEADLESS:-1}" = 0 ]; then . "$SCRIPT_DIR/gui_gpu_env.sh"; fi
set -- --task_graph holding_at --seed "${SEED:-42}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/run.py --task HumanoidTableCarryRsiFT \
 --cfg_train tokenhsi/data/cfg/train/rlg/table_carry_rsi_full_ft.yaml --cfg_env "$OUTPUT_PATH/env.yaml" \
 --motion_file tokenhsi/data/dataset_table_hold_at.yaml --checkpoint "$CKPT" \
 --num_agents 1 --num_envs "${TABLE_TEST_ENVS:-16}" --num_objects 1 \
 --test --eval --eval_repeats "${TABLE_REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 \
 --output_path "$OUTPUT_PATH" --no_video "$@"
