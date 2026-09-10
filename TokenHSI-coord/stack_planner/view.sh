#!/bin/bash
# Local native Isaac Gym window for an isolated stack-planner checkpoint.
# Usage: bash view.sh planner.pth frozen_agent.pth [envs]
set -eo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "usage: bash TokenHSI-coord/stack_planner/view.sh <planner.pth> <frozen-agent.pth> [envs]"
    exit 0
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO="$ROOT/TokenHSI-masteer"
PLANNER=${1:?usage: view.sh <planner.pth> <frozen-agent.pth> [envs]}
POLICY=${2:?usage: view.sh <planner.pth> <frozen-agent.pth> [envs]}
ENVS=${ENVS:-${3:-1}}
GPU=${MA_GPU:-0}
STAGE1=${MS_CKPT:-"$EXEC_REPO/output/ckpt_stage1.pth"}

[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "envs must be positive" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "MA_GPU must be a non-negative integer" >&2; exit 2; }
for file in "$PLANNER" "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 2; }
done
export DISPLAY=${DISPLAY:-:0}
PLANNER=$(realpath -- "$PLANNER")
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

PYTHONPATH="$COORD" python - "$PLANNER" <<'PY'
import sys
from stack_planner.checkpoint import load_stack_checkpoint
_, payload = load_stack_checkpoint(sys.argv[1])
print("stack planner checkpoint: schema={} step={}".format(
    payload["schema_version"], payload.get("step", 0)))
PY

CFG=$(mktemp /tmp/stack_planner_view.XXXXXX.yaml)
cleanup() { rm -f -- "$CFG"; }
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    -e 's/^  enableDebugVis:.*/  enableDebugVis: True/' \
    "$EXEC_REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export STACK_PLANNER_CKPT="$PLANNER"
export STACK_PLANNER_REPLAN_STEPS=${STACK_PLANNER_REPLAN_STEPS:-6}
export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_SCEN=${MS_SCEN:-free} MS_DBG=${MS_DBG:-0}
export MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2} MS_VEL_W=1
export STACK_TASK_MODE=stack STACK_ALLOW_HAND_CONTACT=1
export STACK_CARRY_REHEARSAL_PROB=0 STACK_END_ON_A2_RESUME=0
export STACK_VIRTUAL_RETREAT_BOX=1 STACK_TOP_FOLLOWS_BOTTOM=${STACK_TOP_FOLLOWS_BOTTOM:-1}
export STACK_FIXED_BOX_SIZE_IDS=${STACK_VIEW_BOX_IDS:-${STACK_FIXED_BOX_SIZE_IDS:-}}
export STACK_BOTTOM_DISPLACE_TOL=${STACK_BOTTOM_DISPLACE_TOL:-0.50}
export STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.15}
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-1200}
export STACK_DEBUG=${STACK_DEBUG:-1}

PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

echo "planner: $PLANNER"
echo "agent:   $POLICY"
echo "GPU:     physical $GPU -> logical cuda:0"
echo "DISPLAY: $DISPLAY"
cd "$COORD"
python -u -m stack_planner.run_view \
    --task HumanoidMAStackPlannerView \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --physx --pipeline gpu \
    --cfg_train "$EXEC_REPO/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    --cfg_env "$CFG" \
    --motion_file "$EXEC_REPO/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs "$ENVS" --seed "${STACK_EVAL_SEED:-0}" --test --eval_task carry
