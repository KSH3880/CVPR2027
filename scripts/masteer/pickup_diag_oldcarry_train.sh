#!/bin/bash
# Old carry only with teammate observation zero/live.
# Usage: pickup_diag_oldcarry_train.sh <zero|live> 15000 <gpu> [envs]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODE=${1:?usage: pickup_diag_oldcarry_train.sh <zero|live> 15000 <gpu> [envs]}
TARGET=${2:?usage: pickup_diag_oldcarry_train.sh <zero|live> 15000 <gpu> [envs]}
GPU=${3:?usage: pickup_diag_oldcarry_train.sh <zero|live> 15000 <gpu> [envs]}
ENVS=${4:-2048}
case "$MODE" in zero|live) ;; *) echo "mode must be zero or live" >&2; exit 2 ;; esac
case "$TARGET" in 15000) ;; *) echo "target must be 15000" >&2; exit 2 ;; esac
case "$GPU" in 6|7) ;; *) echo "this diagnostic is restricted to GPU 6 or 7" >&2; exit 2 ;; esac
SHORT=$((TARGET / 1000))k
TAG=pickupdiag_oldcarry_only_teammate_${MODE}_${SHORT}_s0_r1
OUT=$ROOT/runs/output/masteer_pickup/$TAG
CFG=$ROOT/runs/gen_cfgs/masteer_pickup/$TAG.yaml
ENV_FILE=$ROOT/runs/logs/masteer_pickup/$TAG.env
LOG=$ROOT/runs/logs/masteer_pickup/$TAG.log
METRICS=$ROOT/runs/results/masteer_pickup/train_$TAG.npy
BASE=$ROOT/TokenHSI-masteer/output/tokenhsi/ckpt_stage1.pth
TRAIN_SRC=$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml
TRAIN_CFG=$ROOT/runs/gen_cfgs/masteer_pickup/${TAG}_train.yaml
if [ -e "$OUT" ] || [ -e "$LOG" ]; then
    echo "refusing to reuse existing tag: $TAG" >&2
    exit 3
fi
mkdir -p "$(dirname "$OUT")" "$(dirname "$CFG")" "$(dirname "$LOG")" "$(dirname "$METRICS")"
python3 - "$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" "$CFG" "$ENVS" <<'PY'
import pathlib, re, sys
src, dst, envs = sys.argv[1:]
s = pathlib.Path(src).read_text()
s = re.sub(r"^  numAgents:.*$", "  numAgents: 2", s, flags=re.M)
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {envs}", s, flags=re.M)
s = re.sub(r"^  envSpacing:.*$", "  envSpacing: 5", s, flags=re.M)
s = re.sub(r"^(\s*)default_buffer_size_multiplier:",
           r"\1max_gpu_contact_pairs: 4194304\n\1default_buffer_size_multiplier:",
           s, count=1, flags=re.M)
pathlib.Path(dst).write_text(s)
PY
python3 - "$TRAIN_SRC" "$TRAIN_CFG" <<'PY'
import pathlib, re, sys
src, dst = sys.argv[1:]
s = pathlib.Path(src).read_text()
s = re.sub(r"^(\s*)save_frequency:.*$", r"\1save_frequency: 3000", s, flags=re.M)
s = re.sub(r"^(\s*)save_intermediate:.*$", r"\1save_intermediate: True", s, flags=re.M)
pathlib.Path(dst).write_text(s)
PY


export MA_OLD_CARRY_ONLY=1
export MA_TOKEN=$MODE MA_TOKENIZER_ZERO=1 MA_SEP=9 MA_SPAWN_GAP=1.0
export MA_C=0 MA_BETA=0
export MS_SCEN=free MS_MRAND=0 MS_ZERO=0 MS_POS_C=2 MS_VEL_W=1 MS_VEL_K=5
export MS_CLIP=0 MS_ENDCLAMP=0 MS_SEED=0
export MS_PICK_HAND=0.35 MS_PICK_LIFT=0.10 MS_PICK_FRAMES=10
export MA_METRICS=$METRICS MS_METRICS=$METRICS

{
    printf 'export MA_OLD_CARRY_ONLY=%q\n' "$MA_OLD_CARRY_ONLY"
    printf 'export MA_TOKEN=%q\n' "$MA_TOKEN"
    printf 'export MA_TOKENIZER_ZERO=%q\n' "$MA_TOKENIZER_ZERO"
    printf 'export MA_SEP=%q\n' "$MA_SEP"
    printf 'export MA_SPAWN_GAP=%q\n' "$MA_SPAWN_GAP"
    printf 'export MA_C=%q\n' "$MA_C"
    printf 'export MA_BETA=%q\n' "$MA_BETA"
    printf 'export MS_SCEN=%q\n' "$MS_SCEN"
    printf 'export MS_MRAND=%q\n' "$MS_MRAND"
    printf 'export MS_ZERO=%q\n' "$MS_ZERO"
    printf 'export MS_POS_C=%q\n' "$MS_POS_C"
    printf 'export MS_VEL_W=%q\n' "$MS_VEL_W"
    printf 'export MS_VEL_K=%q\n' "$MS_VEL_K"
    printf 'export MS_CLIP=%q\n' "$MS_CLIP"
    printf 'export MS_ENDCLAMP=%q\n' "$MS_ENDCLAMP"
    printf 'export MS_SEED=%q\n' "$MS_SEED"
    printf 'export MS_PICK_HAND=%q\n' "$MS_PICK_HAND"
    printf 'export MS_PICK_LIFT=%q\n' "$MS_PICK_LIFT"
    printf 'export MS_PICK_FRAMES=%q\n' "$MS_PICK_FRAMES"
    printf 'export PICKUP_ENVS=%q\n' "$ENVS"
} > "$ENV_FILE"


cd "$ROOT/TokenHSI-masteer"
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${MA_CONDA_ENV:-tokenhsi_koo}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

python -u ./tokenhsi/run.py \
    --task HumanoidMASteerCarry \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$ENVS" --headless --seed 0 \
    --hrl_checkpoint "$BASE" \
    --max_iterations "$TARGET" --output_path "$OUT" 2>&1 | tee "$LOG"
