#!/bin/bash
# Evaluate one mixed Stage-2 checkpoint and print per-scenario behavior metrics.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"

TAG=${MSTOP_TAG:-mastop_mixed_s0}
GPU=${MSTOP_GPU:-6}
case "$GPU" in
    6|7) ;;
    *) echo "MSTOP_GPU must be physical GPU 6 or 7" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU
ENVS=${MSTOP_ENVS:-512}
SEED=${MSTOP_SEED:-0}
CKPT=${1:-${MSTOP_CKPT:-}}
if [ -z "$CKPT" ]; then
    CKPT=$(find "output/mastop/$TAG" -name Humanoid.pth \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "usage: $0 <stage2-checkpoint> (or set MSTOP_CKPT)" >&2
    exit 2
fi

CFG_SRC=tokenhsi/data/cfg/mastop/amp_humanoid_ma_stop_steer_mixed.yaml
CFG=output/mastop/configs/eval_${TAG}.yaml
TRAIN_CFG=${MSTOP_TRAIN_CFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml}
mkdir -p "$(dirname "$CFG")" output/mastop/metrics
sed "s/^  numEnvs:.*/  numEnvs: $ENVS/" "$CFG_SRC" > "$CFG"

export MSTOP_SCEN=${MSTOP_SCEN:-mixed}
export MSTOP_SEED=$SEED
export MSTOP_MIX=${MSTOP_MIX:-0.35,0.20,0.20,0.25}
export MSTOP_SPEED=${MSTOP_SPEED:-1.5}
export MSTOP_HORIZON=${MSTOP_HORIZON:-2.5}
export MSTOP_SAFE_D=${MSTOP_SAFE_D:-0.9}
export MSTOP_CLEAR=${MSTOP_CLEAR:-1.0}
export MSTOP_PARALLEL_GAP=${MSTOP_PARALLEL_GAP:-2.0}
export MSTOP_NEAR_DT_MIN=${MSTOP_NEAR_DT_MIN:-1.25}
export MSTOP_NEAR_DT_MAX=${MSTOP_NEAR_DT_MAX:-2.50}
export MSTOP_CONFLICT_DT_MAX=${MSTOP_CONFLICT_DT_MAX:-0.35}
export MSTOP_ANGLE_MIN=${MSTOP_ANGLE_MIN:-60}
export MSTOP_ANGLE_MAX=${MSTOP_ANGLE_MAX:-120}
export MSTOP_SPEED_EMA=${MSTOP_SPEED_EMA:-0.20}
export MSTOP_PATH_W=${MSTOP_PATH_W:-0.35}
export MSTOP_VEL_W=${MSTOP_VEL_W:-0.65}
export MSTOP_CREEP_C=${MSTOP_CREEP_C:-0.20}
export MSTOP_CREEP_V=${MSTOP_CREEP_V:-0.10}
export MSTOP_SOLO_RANDOM=1
export MSTOP_METRICS=${MSTOP_METRICS:-output/mastop/metrics/eval_${TAG}.npy}

echo "[mastop] eval checkpoint=$CKPT mix=$MSTOP_MIX envs=$ENVS"
python -u ./tokenhsi/run.py \
    --task HumanoidMAStopSteer \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint "$CKPT" --test --eval --eval_task traj \
    --num_envs "$ENVS" --headless --seed "$SEED"

if [ ! -f "$MSTOP_METRICS" ]; then
    echo "missing metrics: $MSTOP_METRICS" >&2
    exit 3
fi

python - "$MSTOP_METRICS" <<'PY'
import sys
import numpy as np

m = np.load(sys.argv[1])
names = ["free", "solo", "near", "conflict"]
print("MSTOP_EVAL columns: scenario eps colEp colStep dmin stopRate "
      "stopV goV resumed resumeLatency")
for scenario, name in enumerate(names):
    x = m[m[:, 0].astype(int) == scenario]
    if not len(x):
        print(f"MSTOP_EVAL scenario={name} eps=0")
        continue
    stop_steps = x[:, 4:6]
    stop_total = stop_steps.sum()
    stop_v = ((x[:, 9:11] * stop_steps).sum() / stop_total
              if stop_total > 0 else float("nan"))
    go_steps = np.maximum(x[:, 8:9] - stop_steps, 0)
    go_total = go_steps.sum()
    go_v = ((x[:, 11:13] * go_steps).sum() / go_total
            if go_total > 0 else float("nan"))
    latency = x[:, 13]
    latency = latency[latency >= 0]
    latency50 = np.median(latency) if len(latency) else float("nan")
    print(
        f"MSTOP_EVAL scenario={name} eps={len(x)} "
        f"colEp={(x[:, 3] > 0).mean():.4f} "
        f"colStep={x[:, 3].mean():.2f} dmin={np.median(x[:, 2]):.3f} "
        f"stopRate={(stop_steps.sum(axis=1) > 0).mean():.4f} "
        f"stopV={stop_v:.3f} goV={go_v:.3f} "
        f"resumed={(x[:, 7] > 0).mean():.4f} "
        f"resumeLatency={latency50:.3f}")
PY
