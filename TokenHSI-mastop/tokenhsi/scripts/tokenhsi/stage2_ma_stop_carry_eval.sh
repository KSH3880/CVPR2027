#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
ROOT=$(cd "$REPO/.." && pwd)
cd "$REPO"

STAGE=${MSTOP_STAGE:?MSTOP_STAGE=A,B,C,D}
TAG=${MSTOP_TAG:-mastop_eval_${STAGE,,}}
ENV_FILE=${MSTOP_ENV_FILE:-$ROOT/runs/logs/mastop_carry/$TAG.env}
[ ! -f "$ENV_FILE" ] || source "$ENV_FILE"
CKPT=${1:?usage: stage2_ma_stop_carry_eval.sh checkpoint}
GPU=${MSTOP_GPU:-6}
ENVS=${MSTOP_EVAL_ENVS:-512}
case "$STAGE" in A|B|C|D) ;; *) exit 2;; esac
case "$GPU" in 6|7) ;; *) echo "MSTOP_GPU must be 6 or 7" >&2; exit 2;; esac
test -f "$CKPT"
BASE=${MSTOP_BASE_CKPT:-$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth}
CFG_DIR=output/mastop_carry/configs
CFG=$CFG_DIR/eval_${TAG}.yaml
mkdir -p "$CFG_DIR" "$ROOT/runs/results/mastop_carry" "$ROOT/runs/queue/logs"
sed -e "s/^  numAgents:.*/  numAgents: 2/"     -e "s/^  numEnvs:.*/  numEnvs: $ENVS/"     -e "s/^  envSpacing:.*/  envSpacing: 5/"     -e "/default_buffer_size_multiplier:/i\\    max_gpu_contact_pairs: 4194304"     tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml > "$CFG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export MSTOP_STAGE="$STAGE"
export MSTOP_MIX=${MSTOP_EVAL_MIX:-${MSTOP_MIX:-}}
[ -z "$MSTOP_MIX" ] && unset MSTOP_MIX
export MSTOP_SEED=${MSTOP_SEED:-0}
export MS_SEED=${MSTOP_SEED:-0}
export MS_MRAND=${MS_MRAND:-0}
export MS_VEL_W=${MS_VEL_W:-1}
export MS_VEL_K=${MS_VEL_K:-5}
export MS_POS_C=${MS_POS_C:-2}
export MS_ENDCLAMP=${MS_ENDCLAMP:-0}
export MS_CLIP=${MS_CLIP:-0}
if [ "$STAGE" = A ]; then
    export MA_SEP=${MSTOP_MA_SEP:-${MA_SEP:-9}}
else
    export MA_SEP=${MSTOP_MA_SEP:-${MA_SEP:-0}}
fi
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MA_C=${MA_C:-0}
export MA_BETA=${MA_BETA:-0}
export MA_TOKEN=${MA_TOKEN:-live}
export MA_OLD_CARRY_ONLY=${MA_OLD_CARRY_ONLY:-0}
export MA_TEAMMATE_MASK=${MA_TEAMMATE_MASK:-0}
export MSTOP_RESIDUAL_ARCH=${MSTOP_RESIDUAL_ARCH:-0}
export MSTOP_RESIDUAL=${MSTOP_RESIDUAL:-0}
export MSTOP_FROZEN_A=${MSTOP_FROZEN_A:-0}
export MSTOP_RESIDUAL_SCALE=${MSTOP_RESIDUAL_SCALE:-2}
export MSTOP_BRAKE_T=${MSTOP_BRAKE_T:-0.8}
export MSTOP_RESUME_T=${MSTOP_RESUME_T:-0.8}
export MSTOP_STOP_SPEED=${MSTOP_STOP_SPEED:-0.2}
export MSTOP_CARRYWITH=${MSTOP_EVAL_CARRYWITH:-${MSTOP_CARRYWITH:-0}}
export MA_TOKENIZER_ZERO=1
METRICS="$ROOT/runs/results/mastop_carry/eval_${TAG}.npy"
RUNLOG="$ROOT/runs/queue/logs/eval_${TAG}.raw.log"
rm -f "$METRICS"
export MA_METRICS="$METRICS"

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${MSTOP_CONDA_ENV:-tokenhsi_koo}"
python -u ./tokenhsi/run.py     --task HumanoidMAStopCarry     --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml     --cfg_env "$CFG"     --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml     --num_envs "$ENVS" --headless --seed "${MSTOP_SEED:-0}"     --hrl_checkpoint "$BASE" --checkpoint "$CKPT"     --test --eval --eval_task carry > "$RUNLOG" 2>&1
test -f "$METRICS"

python - "$STAGE" "$TAG" "$METRICS" "$ENVS" <<'PY'
import sys
import os
import numpy as np

stage, tag, path, envs = sys.argv[1:]
m = np.load(path)
expected = int(envs) * 2 * 3
if len(m) != expected:
    raise RuntimeError(f"expected {expected} metric rows, got {len(m)}")
if m.shape[1] < 58:
    raise RuntimeError(f"expected 58 metric columns, got {m.shape}")
scenario = m[:, 47].astype(int)
picked = m[:, 40] > 0
finished = m[:, 1] >= 0
print(f"MSTOP_EVAL_SUMMARY stage={stage} tag={tag} eps={len(m)} "
      f"pick={picked.mean():.4f} fin={finished.mean():.4f}")
stats = {}
for sid, name in enumerate("ABCD"):
    x = m[scenario == sid]
    if not len(x):
        continue
    y = x[x[:, 48] > .5]
    stop = y[:, 49] > 0
    steps = y[:, 49].sum()
    hold_steps = y[:, 58].sum() if m.shape[1] > 58 else steps
    brake_steps = y[:, 59].sum() if m.shape[1] > 59 else 0
    resume_steps = y[:, 60].sum() if m.shape[1] > 60 else 0
    stop_rate = stop.mean() if len(stop) else 0.0
    held = y[:, 52].sum() / max(steps, 1)
    root_v = y[:, 50].sum() / max(hold_steps, 1)
    box_v = y[:, 51].sum() / max(hold_steps, 1)
    resumed = y[stop, 54].mean() if stop.any() else 0.0
    collision = (y[:, 56] > 0).mean() if len(y) else 0.0
    stats[sid] = (picked[scenario == sid].mean(),
                  finished[scenario == sid].mean(),
                  stop_rate, held, resumed, root_v, box_v)
    print(f"MSTOP_SCEN_SUMMARY stage={stage} scen={name} eps={len(x)} "
          f"pick={stats[sid][0]:.4f} fin={stats[sid][1]:.4f} "
          f"stopRate={stop_rate:.4f} heldStop={held:.4f} "
          f"rootV={root_v:.3f} boxV={box_v:.3f} resumed={resumed:.4f} "
          f"brakeSteps={brake_steps / max(len(y), 1):.1f} "
          f"resumeSteps={resume_steps / max(len(y), 1):.1f} "
          f"collisionEp={collision:.4f} clear50={np.median(y[:,57]):.3f}")
    if m.shape[1] >= 77:
        z = y[stop]
        strict_steps = y[:, 61].sum() + y[:, 62].sum()
        strict_resume_steps = y[:, 63].sum() + y[:, 64].sum()
        strict_stop = y[:, 61].sum() / max(strict_steps, 1)
        strict_resume = y[:, 63].sum() / max(strict_resume_steps, 1)
        strict_terminal = z[:, 65].mean() if len(z) else 0.0
        strict_retained = ((z[:, 65] > .5) & (z[:, 66] > .5) & (z[:, 69] < .5)).mean() if len(z) else 0.0
        strict_broken = (z[:, 67] > .5).mean() if len(z) else 0.0
        strict_post_drop = (z[:, 69] > .5).mean() if len(z) else 0.0
        if m.shape[1] >= 80:
            contact_stop = y[:, 74].sum() / max(strict_steps, 1)
            contact_resume = y[:, 75].sum() / max(strict_resume_steps, 1)
            contact_terminal = z[:, 76].mean() if len(z) else 0.0
            foot_slip = y[:, 73].sum() / max(y[:, 77].sum(), 1)
            lat_v = y[:, 78].sum() / max(hold_steps, 1)
            yaw_rate = y[:, 79].sum() / max(hold_steps, 1)
        else:
            contact_stop = contact_resume = contact_terminal = float("nan")
            foot_slip = y[:, 73].sum() / max(y[:, 74].sum(), 1)
            lat_v = y[:, 75].sum() / max(hold_steps, 1)
            yaw_rate = y[:, 76].sum() / max(hold_steps, 1)
        print(f"MSTOP_STRICT_SUMMARY stage={stage} scen={name} eps={len(x)} "
              f"bothHandStop={strict_stop:.4f} bothHandResume={strict_resume:.4f} "
              f"contactStop={contact_stop:.4f} contactResume={contact_resume:.4f} "
              f"contactTerminal={contact_terminal:.4f} "
              f"terminal={strict_terminal:.4f} retained={strict_retained:.4f} "
              f"brokenEp={strict_broken:.4f} postStopDropEp={strict_post_drop:.4f} "
              f"footSlip={foot_slip:.3f} latV={lat_v:.3f} yawRate={yaw_rate:.3f}")

if os.environ.get("MSTOP_SKIP_GATE") == "1":
    print(f"MSTOP_GATE_SKIP stage={stage} tag={tag}")
    raise SystemExit(0)

free = stats.get(0)
if free is None or free[0] < .868 or free[1] < .699:
    raise SystemExit("GATE_FAIL free carry regression")
sid = "ABCD".index(stage)
if sid > 0:
    cur = stats.get(sid)
    if cur is None or cur[2] < .5 or cur[3] < .5 or cur[4] < .3 or cur[5] > .6 or cur[6] > .6:
        raise SystemExit(f"GATE_FAIL stage {stage} stop/hold/resume")
print(f"MSTOP_GATE_PASS stage={stage} tag={tag}")
PY
