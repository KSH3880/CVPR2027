#!/bin/bash
# Headless evaluation for the full sequential stack task.
#
# Usage:
#   MA_GPU=7 STACK_EVAL_ENVS=270 \
#     bash scripts/masteer/eval_sequential_stack.sh /abs/Humanoid.pth [tag]
# Default grid sizes are testSizes indices 0,4,7 = 0.22/0.42/0.57 m cubes.
# Override with STACK_EVAL_BOX_SIZE_IDS=i,j,k.
#
# Results:
#   runs/results/sequential_stack/eval_<tag>.npy
#   runs/results/sequential_stack/eval_<tag>.log
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
POLICY=${1:?usage: eval_sequential_stack.sh <policy.pth> [tag]}
TAG=${2:-$(basename "$POLICY" .pth)}
ENVS=${STACK_EVAL_ENVS:-270}
GPU=${MA_GPU:-7}
STAGE1=${MS_CKPT:-"$REPO/output/ckpt_stage1.pth"}
RESULT_DIR="$ROOT/runs/results/sequential_stack"
METRICS="$RESULT_DIR/eval_${TAG}.npy"
LOG="$RESULT_DIR/eval_${TAG}.log"

case "$TAG" in
    ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2 ;;
esac
case "$GPU" in
    0|1|2|3|4|5|6|7) ;;
    *) echo "MA_GPU는 physical GPU index 0..7이어야 한다: $GPU" >&2; exit 2 ;;
esac
case "$ENVS" in
    ""|*[!0-9]*) echo "STACK_EVAL_ENVS는 양의 정수여야 한다: $ENVS" >&2; exit 2 ;;
esac
[ "$ENVS" -gt 0 ] || exit 2
BOX_GRID=${STACK_EVAL_BOX_GRID:-1}
if [ "$BOX_GRID" != 0 ] && [ $((ENVS % 9)) -ne 0 ]; then
    echo "3x3 box grid에서는 STACK_EVAL_ENVS가 9의 배수여야 한다: $ENVS" >&2
    exit 2
fi
for file in "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 1; }
done
[ ! -e "$METRICS" ] && [ ! -e "$LOG" ] || {
    echo "기존 평가 결과를 덮어쓰지 않는다. 다른 tag를 사용한다: $TAG" >&2
    exit 1
}
mkdir -p "$RESULT_DIR"

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    else
        CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || {
    echo "conda 초기화 파일 없음: $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
}
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

CFG=$(mktemp /tmp/sequential_stack_eval.XXXXXX.yaml)
cleanup() { rm -f -- "$CFG"; }
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    "$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
    > "$CFG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_METRICS="$METRICS"
export MA_EVAL_ALL_AGENT_ROWS=1
export MA_EVAL_FLUSH_DONE=1
export STACK_EVAL_BOX_GRID="$BOX_GRID"
export STACK_EVAL_BOX_SIZE_IDS=${STACK_EVAL_BOX_SIZE_IDS:-0,4,7}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=${MS_CLIP:-1}
export MS_REWARD_OUTER=${MS_REWARD_OUTER:-1}
export MS_POS_C=${MS_POS_C:-1.2}
export MS_VEL_W=${MS_VEL_W:-1}
export MS_SCEN=${MS_SCEN:-free}
export STACK_TASK_MODE=stack
export STACK_CARRY_REHEARSAL_PROB=0
export STACK_END_ON_A2_RESUME=0
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-900}
export STACK_BOTTOM_Z_TOL=${STACK_BOTTOM_Z_TOL:-0.05}

echo "sequential-stack eval: policy=$POLICY envs=$ENVS gpu=$GPU box_grid=$STACK_EVAL_BOX_GRID size_ids=$STACK_EVAL_BOX_SIZE_IDS" | tee "$LOG"
cd "$REPO"
python -u ./tokenhsi/run.py \
    --task HumanoidMASequentialStackCarry \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 \
    --physx --pipeline gpu --headless \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs "$ENVS" --seed "${STACK_EVAL_SEED:-0}" \
    --test --eval --eval_task carry 2>&1 | tee -a "$LOG"

[ -f "$METRICS" ] || { echo "metric 파일이 생성되지 않았다: $METRICS" >&2; exit 5; }
python3 - "$TAG" "$METRICS" "$ENVS" <<'PY' | tee -a "$LOG"
import sys
import numpy as np

tag, path, envs = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = np.load(path)
if m.ndim != 2 or m.shape[1] < 54:
    raise SystemExit(f"sequential metric columns >=54 expected, got {m.shape}")

# The stock player performs three repeats; repeat 0 is an unsettled warm-up.
# Each repeat completes `num_envs` agent rows.  Keep repeats 1 and 2 only.
rows_per_repeat = 2 * envs
if len(m) >= 3 * rows_per_repeat:
    m = m[rows_per_repeat:3 * rows_per_repeat]
    warmup = "discarded"
else:
    warmup = "unavailable"

# Pair the kth A1 row with the kth A2 row from the same environment.  Overall
# path error covers both agents; place/retreat use A1 and top placement uses A2.
by_row = {}
for row in m:
    by_row.setdefault(int(row[0]), []).append(row)
episodes = []
for env in sorted({row_id // 2 for row_id in by_row}):
    a1, a2 = by_row.get(2 * env, []), by_row.get(2 * env + 1, [])
    for index in range(min(len(a1), len(a2))):
        episodes.append((a1[index], a2[index]))
if not episodes:
    raise SystemExit("no paired sequential-stack episodes in metrics")

a1 = np.stack([pair[0] for pair in episodes])
a2 = np.stack([pair[1] for pair in episodes])
success = a1[:, 40] > 0.5
root_sum = a1[:, 26] + a2[:, 26]
box_sum = a1[:, 27] + a2[:, 27]
steps = np.maximum(a1[:, 30] + a2[:, 30], 1.0)

def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")

def stats(prefix, mask):
    if not np.any(mask):
        return f" {prefix}_root_cum=NA {prefix}_box_cum=NA {prefix}_root_mae=NA {prefix}_box_mae=NA"
    return (f" {prefix}_root_cum={mean(root_sum[mask]):.4f}"
            f" {prefix}_box_cum={mean(box_sum[mask]):.4f}"
            f" {prefix}_root_mae={mean(root_sum[mask] / steps[mask]):.4f}"
            f" {prefix}_box_mae={mean(box_sum[mask] / steps[mask]):.4f}")

parts = [
    f"SEQ_STACK_EVAL tag={tag}",
    f"episodes={len(episodes)}",
    f"success_n={int(success.sum())}",
    f"success_rate={success.mean():.4f}",
    f"warmup={warmup}",
    f"root_cum={mean(root_sum):.4f}",
    f"box_cum={mean(box_sum):.4f}",
    f"root_mae={mean(root_sum / steps):.4f}",
    f"box_mae={mean(box_sum / steps):.4f}",
]

for name, source, col in (("place", a1, 42),
                          ("retreat", a1, 45),
                          ("a2", a2, 48)):
    n = np.maximum(source[:, col + 2], 1.0)
    parts.extend((
        f"{name}_root_cum={mean(source[:, col]):.4f}",
        f"{name}_box_cum={mean(source[:, col + 1]):.4f}",
        f"{name}_root_mae={mean(source[:, col] / n):.4f}",
        f"{name}_box_mae={mean(source[:, col + 1] / n):.4f}",
    ))

print(" ".join(parts) + stats("success", success))

def size_label(row):
    return "x".join(f"{value:.2f}" for value in row[51:54])

combo_keys = sorted({(size_label(a1[i]), size_label(a2[i]))
                     for i in range(len(episodes))})
for bottom, top in combo_keys:
    mask = np.array([(size_label(a1[i]) == bottom
                      and size_label(a2[i]) == top)
                     for i in range(len(episodes))])
    print(
        f"SEQ_STACK_BOX_COMBO tag={tag} bottom={bottom} top={top} "
        f"episodes={int(mask.sum())} success_n={int(success[mask].sum())} "
        f"success_rate={mean(success[mask]):.4f} "
        f"root_cum={mean(root_sum[mask]):.4f} "
        f"box_cum={mean(box_sum[mask]):.4f} "
        f"root_mae={mean(root_sum[mask] / steps[mask]):.4f} "
        f"box_mae={mean(box_sum[mask] / steps[mask]):.4f}")
PY

echo "metrics: $METRICS"
echo "log:     $LOG"
