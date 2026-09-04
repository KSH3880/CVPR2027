#!/bin/bash
# Train on any number of GPUs (1, 2, 4, 8) with the original learning recipe preserved.
#
#   bash train.sh 1 tokenhsi/scripts/single_task/traj_train.sh
#   bash train.sh 8 tokenhsi/scripts/tokenhsi/stage1_train.sh
#   bash train.sh 8 tokenhsi/scripts/single_task/traj_train.sh --fast
#   bash train.sh 4 tokenhsi/scripts/single_task/sit_train.sh --dry-run
#
# Default mode splits the batch across GPUs, so 1/2/4/8 GPUs all train the same
# way as the original single-GPU recipe - only faster. num_envs, minibatch_size
# and amp_minibatch_size are each divided by the GPU count, which keeps the
# per-update batch and the number of updates identical.
#
#   --fast     give every GPU the script's full num_envs instead. N times the
#              data per epoch, so the batch grows N times - faster, but no
#              longer the same recipe (raise learning_rate to compensate).
#   --dry-run  print the resolved settings and exit.
#
# Runs against the TokenHSI baseline by default. Pick another repo with TOKENHSI_REPO:
#   TOKENHSI_REPO=TokenHSI-steer bash train.sh 4 tokenhsi/scripts/single_task/carry_train.sh
set -e

N=${1:?usage: train.sh <num_gpus> <train_script.sh> [--fast] [--dry-run] [extra args]}
SCRIPT=${2:?usage: train.sh <num_gpus> <train_script.sh> [--fast] [--dry-run] [extra args]}
shift 2

MODE=same
DRY=0
EXTRA=()
for a in "$@"; do
    case "$a" in
        --fast)    MODE=fast ;;
        --dry-run) DRY=1 ;;
        *)         EXTRA+=("$a") ;;
    esac
done

REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d /home/hwanhee/CVPR2027/$REPO/tokenhsi ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd /home/hwanhee/CVPR2027/$REPO

CFG=$(grep -oE '\-\-cfg_train [^ \\]+' "$SCRIPT" | awk '{print $2}')
ENVS=$(grep -oE '\-\-num_envs [0-9]+' "$SCRIPT" | awk '{print $2}')
ARGS=$(grep -v '^#!' "$SCRIPT" | tr '\n' ' ' | sed 's/\\/ /g' \
       | sed 's|^ *python *\./tokenhsi/run\.py||' \
       | sed -E 's/--num_envs [0-9]+//' | sed -E 's|--cfg_train [^ ]+||')

GENDIR=/home/hwanhee/CVPR2027/runs/gen_cfgs/$REPO
mkdir -p $GENDIR
NEWCFG=$GENDIR/$(basename "$SCRIPT" .sh)_${N}gpu_${MODE}.yaml

if [ "$MODE" = same ] && [ "$N" -gt 1 ]; then
    NEWENVS=$(python - "$CFG" "$ENVS" "$N" "$NEWCFG" <<'PY'
import re, sys
cfg, envs, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
text = open(cfg).read()
def scale(key):
    m = re.search(rf"^(\s*){key}: (\d+)", text, re.M)
    v = int(m.group(2))
    if v % n:
        sys.exit(f"{key}={v} is not divisible by {n} GPUs")
    return v // n
if envs % n:
    sys.exit(f"num_envs={envs} is not divisible by {n} GPUs")
for key in ("minibatch_size", "amp_minibatch_size"):
    new = scale(key)
    text = re.sub(rf"^(\s*){key}: \d+", rf"\g<1>{key}: {new}", text, flags=re.M)
open(out, "w").write(text)
print(envs // n)
PY
)
else
    cp "$CFG" "$NEWCFG"
    NEWENVS=$ENVS
fi

MB=$(grep -oE '^\s*minibatch_size: [0-9]+' "$NEWCFG" | grep -oE '[0-9]+')
AMB=$(grep -oE '^\s*amp_minibatch_size: [0-9]+' "$NEWCFG" | grep -oE '[0-9]+')
echo "=============================================================="
echo " $(basename $SCRIPT)   ${N} GPU   mode=${MODE}"
echo " num_envs        ${ENVS} -> ${NEWENVS} per GPU  (total $((NEWENVS * N)))"
echo " minibatch_size  ${MB} per GPU   (global $((MB * N)))"
echo " amp_minibatch   ${AMB} per GPU   (global $((AMB * N)))"
echo "=============================================================="
[ "$DRY" = 1 ] && exit 0

export PYTHONPATH=/home/hwanhee/CVPR2027/ddp_shim:$PYTHONPATH

if [ "$N" = 1 ]; then
    exec python ./tokenhsi/run.py $ARGS --cfg_train "$NEWCFG" --num_envs $NEWENVS "${EXTRA[@]}"
else
    exec torchrun --standalone --nproc_per_node=$N ./tokenhsi/run.py $ARGS \
        --cfg_train "$NEWCFG" --num_envs $NEWENVS --horovod "${EXTRA[@]}"
fi
