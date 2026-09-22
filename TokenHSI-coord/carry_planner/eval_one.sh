#!/bin/bash
# Deterministic multi-environment evaluation for one plain-Carry planner.
# Usage: eval_one.sh <planner.pth> <frozen-ms18.pth> <mixed|converge|cross|free> [envs] [seed]
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO=${CARRY_PLANNER_EXEC_REPO:-"$ROOT/TokenHSI-masteer"}
PLANNER=${1:?usage: eval_one.sh <planner.pth> <frozen-ms18.pth> <mixed|converge|cross|free> [envs] [seed]}
POLICY=${2:?usage: eval_one.sh <planner.pth> <frozen-ms18.pth> <mixed|converge|cross|free> [envs] [seed]}
PROFILE=${3:?usage: eval_one.sh <planner.pth> <frozen-ms18.pth> <mixed|converge|cross|free> [envs] [seed]}
ENVS=${4:-64}
SEED=${5:-0}
GPU=${MA_GPU:-0}
STAGE1=${MS_CKPT:-"$EXEC_REPO/output/tokenhsi/ckpt_stage1.pth"}

case "$PROFILE" in
    mixed) SCENE=cross; CONVERGE=0.75 ;;
    converge) SCENE=cross; CONVERGE=1.0 ;;
    cross) SCENE=cross; CONVERGE=0.0 ;;
    free) SCENE=free; CONVERGE=0.0 ;;
    *) echo "profile must be mixed, converge, cross, or free" >&2; exit 2 ;;
esac
[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "envs must be positive" >&2; exit 2; }
[[ "$SEED:$GPU" =~ ^[0-9]+:[0-9]+$ ]] || { echo "seed/GPU must be non-negative" >&2; exit 2; }
for file in "$PLANNER" "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 2; }
done
PLANNER=$(realpath -- "$PLANNER")
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || {
    echo "NVML physical GPU index $GPU is not available" >&2; exit 2;
}
case "$GPU_UUID" in GPU-*|MIG-*) ;; *) echo "invalid GPU UUID: $GPU_UUID" >&2; exit 2;; esac

NAME=$(basename "$PLANNER" .pth)
RUN=$(basename "$(dirname "$PLANNER")")
SAFE_RUN=$(printf '%s' "$RUN" | tr -c 'A-Za-z0-9_.-' '_')
EVAL_ID="${NAME}_${PROFILE}_s${SEED}"
OUT="$ROOT/runs/results/carry_planner/$SAFE_RUN"
LOG="$OUT/$EVAL_ID.log"
METRICS="$OUT/$EVAL_ID.npy"
SUMMARY="$OUT/$EVAL_ID.json"
LOCK="$OUT/$EVAL_ID.lock"
mkdir -p "$OUT"
if ! mkdir "$LOCK" 2>/dev/null; then echo "이미 평가 중: $EVAL_ID" >&2; exit 4; fi
TMP=$(mktemp -d "/tmp/carry_planner_eval.XXXXXX")
CFG="$TMP/env.yaml"
cleanup() { rm -rf -- "$LOCK" "$TMP"; }
trap cleanup EXIT INT TERM
mkdir -p "$TMP/nn"
cp -- "$POLICY" "$TMP/nn/Humanoid.pth"
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    -e 's/^  enableDebugVis:.*/  enableDebugVis: False/' \
    "$COORD/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"
rm -f -- "$METRICS" "$SUMMARY"

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

export CARRY_PLANNER_PHYSICAL_GPU="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID"
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0
export CARRY_PLANNER_CKPT="$PLANNER" CARRY_PLANNER_DEBUG=0
export CARRY_PLANNER_REPLAN_STEPS=${CARRY_PLANNER_REPLAN_STEPS:-6}
export CARRY_PLANNER_CONVERGE_PROB="$CONVERGE"
export CARRY_PLANNER_GOAL_MARGIN=${CARRY_PLANNER_GOAL_MARGIN:-0.25}
export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_SEP=${MA_SEP:-0} MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_SCEN="$SCENE" MS_SEED="$SEED" MS_DBG=0
export MS_VIEW_TIMED_CROSS=0 COORD_VIEWER=0
export MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2} MS_VEL_W=1
export COORD_SPEED_LIMIT=1 COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75}
export COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_ALLOW_HAND_CONTACT=${COORD_ALLOW_HAND_CONTACT:-1}
export COORD_PRESERVE_PICKUP_APPROACH=${COORD_PRESERVE_PICKUP_APPROACH:-1}
export MS_METRICS="$METRICS" MA_METRICS="$METRICS"
unset MA_LAYOUT MA_LAYOUT_D MA_LAYOUT_S MA_LAYOUT_L MS_SCEN_CURVE MS_VIZ MA_VIDEO

. "$COORD/stack_planner/physx_cuda_compat.sh"
echo "carry eval: run=$RUN ckpt=$NAME profile=$PROFILE envs=$ENVS seed=$SEED physical_gpu=$GPU"
cd "$COORD"
set +e
python -u -m carry_planner.run_view \
    --task HumanoidMACarryPlannerView --headless \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$TMP/nn/Humanoid.pth" \
    --num_envs "$ENVS" --seed "$SEED" --test --eval --eval_task carry > "$LOG" 2>&1
RC=$?
set -e
if [ "$RC" -ne 0 ] || [ ! -f "$METRICS" ]; then
    echo "CARRY_PLANNER_EVAL_ERROR id=$EVAL_ID rc=$RC metrics=$([ -f "$METRICS" ] && echo yes || echo no)" | tee -a "$LOG" >&2
    [ "$RC" -eq 0 ] || exit "$RC"
    exit 5
fi
python "$ROOT/scripts/coord/summarize_eval.py" \
    --tag "$SAFE_RUN/$NAME" --scenario "$PROFILE" --checkpoint "$PLANNER" \
    --log "$LOG" --metrics "$METRICS" --output "$SUMMARY" | tee -a "$LOG"
