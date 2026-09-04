#!/bin/bash
# Strict floor-start pickup evaluation for f22x2 or pickupdiag_* checkpoints.
# Usage: pickup_diag_eval.sh <f22x2|tag> <gpu> [envs]
# Set PICKUP_ITER=3000 to select Humanoid_00003000.pth instead of Humanoid.pth.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=${1:?usage: pickup_diag_eval.sh <f22x2|tag> <gpu> [envs]}
GPU=${2:?usage: pickup_diag_eval.sh <f22x2|tag> <gpu> [envs]}
ENVS=${3:-512}
SUFFIX=${PICKUP_EVAL_SUFFIX:-fresh}
RUN=${NAME}_${SUFFIX}
CFG=$ROOT/runs/gen_cfgs/masteer_pickup/eval_$RUN.yaml
LOG=$ROOT/runs/logs/masteer_pickup/eval_$RUN.log
METRICS=$ROOT/runs/results/masteer_pickup/eval_$RUN.npy
BASE=$ROOT/TokenHSI-masteer/output/tokenhsi/ckpt_stage1.pth
TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml

case "$GPU" in 6|7) ;; *) echo "this diagnostic is restricted to GPU 6 or 7" >&2; exit 2 ;; esac
if [ -e "$LOG" ] || [ -e "$METRICS" ]; then
    echo "evaluation already exists: $RUN (set a new PICKUP_EVAL_SUFFIX)" >&2
    exit 3
fi
mkdir -p "$(dirname "$CFG")" "$(dirname "$LOG")" "$(dirname "$METRICS")"

if [ "$NAME" = f22x2 ]; then
    TASK=HumanoidMAF22Carry
    CK=$ROOT/runs/checkpoints/steer/f22b_long_c20/Humanoid_00025000.pth
    SRC=$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml
    export MA_TOKENIZER_ZERO=1 MA_SEP=9 MA_SPAWN_GAP=1.0
    export MA_C=0 MA_BETA=0
    export MS_SCEN=free MS_MRAND=0 MS_ZERO=0 MS_POS_C=2 MS_VEL_W=1 MS_VEL_K=5
    export MS_CLIP=0 MS_ENDCLAMP=0 MS_SEED=0
    export MS_PICK_HAND=0.35 MS_PICK_LIFT=0.10 MS_PICK_FRAMES=10
else
    TASK=HumanoidMASteerCarry
    ENV_FILE=$ROOT/runs/logs/masteer_pickup/$NAME.env
    SRC=$ROOT/runs/gen_cfgs/masteer_pickup/$NAME.yaml
    OUT=$ROOT/runs/output/masteer_pickup/$NAME
    if [ ! -f "$ENV_FILE" ] || [ ! -f "$SRC" ]; then
        echo "missing sidecar or training config for $NAME" >&2
        exit 4
    fi
    if [ ! -d "$OUT" ]; then
        echo "missing training output for $NAME: $OUT" >&2
        exit 4
    fi
    source "$ENV_FILE"
    if [ -n "${PICKUP_ITER:-}" ]; then
        case "$PICKUP_ITER" in
            *[!0-9]*) echo "PICKUP_ITER must be an integer: $PICKUP_ITER" >&2; exit 2 ;;
        esac
        printf -v CK_NAME 'Humanoid_%08d.pth' "$PICKUP_ITER"
        CK=$(find "$OUT" -name "$CK_NAME" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    else
        CK=$(find "$OUT" -name Humanoid.pth -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
fi
if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
    echo "missing checkpoint: ${CK:-unset}" >&2
    exit 4
fi

python3 - "$SRC" "$CFG" "$ENVS" <<'PY'
import pathlib, re, sys
src, dst, envs = sys.argv[1:]
s = pathlib.Path(src).read_text()
s = re.sub(r"^  numAgents:.*$", "  numAgents: 2", s, flags=re.M)
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {envs}", s, flags=re.M)
pathlib.Path(dst).write_text(s)
PY

export MA_METRICS=$METRICS MS_METRICS=$METRICS
cd "$ROOT/TokenHSI-masteer"
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${MA_CONDA_ENV:-tokenhsi_koo}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

set +e
python -u ./tokenhsi/run.py \
    --task "$TASK" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$ENVS" --headless --seed 0 \
    --hrl_checkpoint "$BASE" --checkpoint "$CK" \
    --test --eval --eval_task carry 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ] || [ ! -f "$METRICS" ]; then
    echo "pickup eval failed: rc=$RC metrics=$([ -f "$METRICS" ] && echo yes || echo no)" >&2
    exit "$([ "$RC" -ne 0 ] && echo "$RC" || echo 5)"
fi
python3 "$ROOT/scripts/masteer/pickup_diag_summary.py" "$RUN" "$LOG" "$METRICS" 2 | tee -a "$LOG"
