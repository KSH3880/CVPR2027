#!/bin/bash
# Native Isaac Gym viewer for the plain Carry planner.
# Usage: view.sh <planner.pth> <frozen-ms18.pth>
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO=${CARRY_PLANNER_EXEC_REPO:-"$ROOT/TokenHSI-masteer"}
PLANNER=${1:?usage: view.sh <planner.pth> <frozen-ms18.pth>}
POLICY=${2:?usage: view.sh <planner.pth> <frozen-ms18.pth>}
GPU=${MA_GPU:-0}
STAGE1=${MS_CKPT:-"$EXEC_REPO/output/ckpt_stage1.pth"}

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "MA_GPU must be non-negative" >&2; exit 2; }
for file in "$PLANNER" "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 2; }
done
PLANNER=$(realpath -- "$PLANNER")
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export TOKENHSI_GRAPHICS_DEVICE_ID=${TOKENHSI_GRAPHICS_DEVICE_ID:-$GPU}
export DISPLAY=${DISPLAY:-:0}

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
model, payload = load_stack_checkpoint(sys.argv[1])
if not model.config.plain_carry:
    raise SystemExit("checkpoint is not a plain-carry planner")
print("carry planner checkpoint: schema={} step={} history={}".format(
    payload["schema_version"], payload.get("step", 0),
    model.config.history_steps))
PY

CFG=$(mktemp /tmp/carry_planner_view.XXXXXX.yaml)
cleanup() { rm -f -- "$CFG"; }
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e 's/^  numEnvs:.*/  numEnvs: 1/' \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    -e 's/^  enableDebugVis:.*/  enableDebugVis: False/' \
    "$COORD/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0
export CARRY_PLANNER_CKPT="$PLANNER"
export CARRY_PLANNER_REPLAN_STEPS=${CARRY_PLANNER_REPLAN_STEPS:-6}
export CARRY_PLANNER_DEBUG=${CARRY_PLANNER_DEBUG:-1}
export CARRY_PLANNER_CONVERGE_PROB=${CARRY_PLANNER_CONVERGE_PROB:-0.75}
export CARRY_PLANNER_GOAL_MARGIN=${CARRY_PLANNER_GOAL_MARGIN:-0.25}
export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_SEP=${MA_SEP:-0} MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_SCEN=${MS_SCEN:-cross}
# The ordinary Cross layout only guarantees intersecting routes.  For the
# viewer, align the two nominal arrival times as well so a straight/full-speed
# execution is a genuine collision case.  Keep free/other layouts usable by
# disabling this default outside Cross, and allow an explicit caller override.
export COORD_VIEWER=1
if [ -z "${MS_VIEW_TIMED_CROSS+x}" ]; then
    if [ "$MS_SCEN" = cross ]; then MS_VIEW_TIMED_CROSS=1
    else MS_VIEW_TIMED_CROSS=0
    fi
fi
export MS_VIEW_TIMED_CROSS
export MS_VIEW_TIMED_CROSS_PROB=${MS_VIEW_TIMED_CROSS_PROB:-1.0}
export MS_VIEW_TIMED_CROSS_TOL=${MS_VIEW_TIMED_CROSS_TOL:-0.25}
export MS_VIEW_TIMED_CROSS_PRE=${MS_VIEW_TIMED_CROSS_PRE:-2.0}
export MS_VIEW_TIMED_CROSS_POST=${MS_VIEW_TIMED_CROSS_POST:-4.0}
export MS_VIEW_TIMED_CROSS_MAX_SHIFT=${MS_VIEW_TIMED_CROSS_MAX_SHIFT:-8.0}
export MS_DBG=${MS_DBG:-1} MS_DRAW_SPEED=${MS_DRAW_SPEED:-1}
export MS_CAM=${MS_CAM:-top} MS_CAM_H=${MS_CAM_H:-17} MS_CAM_B=${MS_CAM_B:-9}
export MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2} MS_VEL_W=1
export COORD_SPEED_LIMIT=1 COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75}
export COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_ALLOW_HAND_CONTACT=${COORD_ALLOW_HAND_CONTACT:-1}
export COORD_PRESERVE_PICKUP_APPROACH=${COORD_PRESERVE_PICKUP_APPROACH:-1}

. "$COORD/stack_planner/physx_cuda_compat.sh"

echo "planner:  $PLANNER"
echo "executor: $POLICY"
echo "scene:    $MS_SCEN  timed_cross=$MS_VIEW_TIMED_CROSS  replan=$CARRY_PLANNER_REPLAN_STEPS"
echo "GPU:      physical $GPU -> logical cuda:0"
echo "DISPLAY:  $DISPLAY"

cd "$COORD"
python -u -m carry_planner.run_view \
    --task HumanoidMACarryPlannerView \
    --sim_device cuda:0 --rl_device cuda:0 \
    --graphics_device_id "$TOKENHSI_GRAPHICS_DEVICE_ID" --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs 1 --seed "${CARRY_PLANNER_VIEW_SEED:-0}" \
    --test --eval --eval_task carry
