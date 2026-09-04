#!/bin/bash
# Final evaluation for one MA training row. Replays the row's saved environment
# and requires both the frozen stage-1 checkpoint and the trained adapt policy.
set -u

ROOT=/home/hwanhee/CVPR2027
TAG=${1:?usage: eval_one.sh <tag> <gpu> [envs]}
GPU=${2:?usage: eval_one.sh <tag> <gpu> [envs]}
ENVS=${3:-512}
CLAIM=$ROOT/runs/queue/gpu_locks/eval_$TAG
LOG=$ROOT/runs/queue/logs/eval_$TAG.log
ENV_FILE=$ROOT/runs/queue/logs/$TAG.env

if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "ma eval: $TAG is already claimed" >&2
    exit 3
fi
trap 'rm -rf "$CLAIM"' EXIT
printf '%s\n' "$GPU" > "$CLAIM/gpu"

if [ ! -f "$ENV_FILE" ]; then
    echo "ma eval: missing $ENV_FILE; refusing to guess training settings" >&2
    exit 4
fi
source "$ENV_FILE"

CK=$(python3 - "$ROOT/TokenHSI-ma/output/ma/$TAG" <<'PY'
import sys
from pathlib import Path

paths = list(Path(sys.argv[1]).glob("*/nn/Humanoid.pth"))
if paths:
    print(max(paths, key=lambda path: path.stat().st_mtime))
PY
)
TRAIN_ENV=$ROOT/runs/gen_cfgs/ma/$TAG.yaml
if [ -z "$CK" ] || [ ! -f "$TRAIN_ENV" ]; then
    echo "ma eval: missing checkpoint or generated env config for $TAG" >&2
    exit 4
fi

EVAL_ENV=$ROOT/runs/gen_cfgs/ma/eval_$TAG.yaml
python3 - "$TRAIN_ENV" "$EVAL_ENV" "$ENVS" <<'PY'
import re, sys

text = open(sys.argv[1]).read()
text = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[3]}", text, flags=re.M)
open(sys.argv[2], "w").write(text)
PY

TRAIN_CFG=${MA_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml}
if [ -f "$ROOT/runs/gen_cfgs/ma/${TAG}_train.yaml" ]; then
    TRAIN_CFG=$ROOT/runs/gen_cfgs/ma/${TAG}_train.yaml
fi
BASE_CKPT=${MA_CKPT:-output/tokenhsi/ckpt_stage1.pth}
METRICS=$ROOT/runs/results/ma/eval_$TAG.npy
mkdir -p "$(dirname "$METRICS")"
rm -f "$METRICS"

cd "$ROOT/TokenHSI-ma"
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
export CUDA_VISIBLE_DEVICES=$GPU
export MA_METRICS=$METRICS

python -u ./tokenhsi/run.py \
    --task "${MA_TASK:-HumanoidMACarry}" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$EVAL_ENV" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$ENVS" --headless --seed "${MA_SEED:-0}" \
    --hrl_checkpoint "$BASE_CKPT" --checkpoint "$CK" \
    --test --eval --eval_task "${MA_EVAL_TASK:-carry}" > "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$METRICS" ]; then
    printf 'MA_EVAL_ERROR tag=%s rc=%s metrics=%s\n' "$TAG" "$rc" "$([ -f "$METRICS" ] && echo yes || echo no)" >> "$LOG"
    [ "$rc" -ne 0 ] && exit "$rc"
    exit 5
fi

python3 - "$TAG" "$LOG" "$METRICS" "${MA_AGENTS:-2}" >> "$LOG" <<'PY'
import re, sys
import numpy as np

tag, log_path, metrics_path, agents = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
text = open(log_path, errors="ignore").read()
rates = [float(value) for value in re.findall(r"'success_rate': ([0-9.]+)", text)]
stable = rates[1:] if len(rates) > 1 else rates
sr = sum(stable) / len(stable) if stable else float("nan")
m = np.load(metrics_path)
finished = m[:, 1] >= 0
agent = m[:, 0].astype(int) % agents
paths = [np.median(m[agent == index, 5]) for index in range(agents)]
fmt = lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
print(
    f"MA_EVAL_SUMMARY tag={tag} rc=0 sr={fmt(sr)} eps={len(m)} "
    f"fin={finished.mean():.4f} "
    f"t50={np.median(m[finished, 1]) if finished.any() else -1:.0f} "
    f"colEp={(m[:, 2] > 0).mean():.4f} colStep={m[:, 2].mean():.2f} "
    f"enter={m[:, 3].mean():.2f} still={m[:, 4].mean():.2f} "
    f"path={np.median(m[:, 5]):.2f} "
    f"pathA={','.join(f'{value:.2f}' for value in paths)}"
)
PY

grep '^MA_EVAL_SUMMARY ' "$LOG" | tail -1
