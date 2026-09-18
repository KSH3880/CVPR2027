#!/usr/bin/env bash
# Visualize the E2E DAgger planner in the native Isaac Gym window.
set -euo pipefail

TARGET_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=${E2E_SOURCE_ROOT:-/home/visitor/hyunwoo/CVPR2027}
PLANNER=${E2E_CKPT:-}
POLICY=${E2E_POLICY:-}
STAGE1=${MS_CKPT:-}
ENVS=${ENVS:-1}
GPU=${MA_GPU:-0}
GRAPHICS=${TOKENHSI_GRAPHICS_DEVICE_ID:-}
SEED=${STACK_EVAL_SEED:-0}
CHECK_ONLY=0

usage() {
    cat <<'EOF'
usage: ./visualize_e2e_planner.sh [options]

Visualizes the slide's best model by default:
  e2e_planner m4_bc_r5/last.pth + E2E_CARRY_END_OFFSET=0.15

Options:
  --planner PATH       e2e planner checkpoint
  --policy PATH        frozen sequential-stack executor checkpoint
  --source-root PATH   checkout containing e2e_planner and TokenHSI-masteer
  --envs N             number of scenes (default: 1)
  --gpu N              physical CUDA GPU (default: MA_GPU or 0)
  --graphics N         Vulkan graphics device (default: same as GPU)
  --seed N             simulator seed (default: 0)
  --check              validate files and checkpoint without opening a window
  -h, --help           show this help

Environment overrides remain available, including DISPLAY, MS_CAM, MS_CAM_H,
MS_CAM_B, E2E_REPLAN_STEPS, and STACK_EPISODE_LENGTH.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --planner)
            [ $# -ge 2 ] || { echo "--planner requires a path" >&2; exit 2; }
            PLANNER=$2
            shift 2
            ;;
        --policy)
            [ $# -ge 2 ] || { echo "--policy requires a path" >&2; exit 2; }
            POLICY=$2
            shift 2
            ;;
        --source-root)
            [ $# -ge 2 ] || { echo "--source-root requires a path" >&2; exit 2; }
            SOURCE_ROOT=$2
            shift 2
            ;;
        --envs)
            [ $# -ge 2 ] || { echo "--envs requires a value" >&2; exit 2; }
            ENVS=$2
            shift 2
            ;;
        --gpu)
            [ $# -ge 2 ] || { echo "--gpu requires a value" >&2; exit 2; }
            GPU=$2
            shift 2
            ;;
        --graphics)
            [ $# -ge 2 ] || { echo "--graphics requires a value" >&2; exit 2; }
            GRAPHICS=$2
            shift 2
            ;;
        --seed)
            [ $# -ge 2 ] || { echo "--seed requires a value" >&2; exit 2; }
            SEED=$2
            shift 2
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SOURCE_ROOT=$(realpath -- "$SOURCE_ROOT")
PLANNER=${PLANNER:-"$SOURCE_ROOT/runs/e2e_planner/bc/m4_bc_r5/last.pth"}
POLICY=${POLICY:-"$SOURCE_ROOT/TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00011000.pth"}
STAGE1=${STAGE1:-"$SOURCE_ROOT/TokenHSI-masteer/output/ckpt_stage1.pth"}
# Explicit relative paths may be relative to the selected source checkout.
if [ ! -f "$PLANNER" ] && [ -f "$SOURCE_ROOT/$PLANNER" ]; then PLANNER="$SOURCE_ROOT/$PLANNER"; fi
if [ ! -f "$POLICY" ] && [ -f "$SOURCE_ROOT/$POLICY" ]; then POLICY="$SOURCE_ROOT/$POLICY"; fi
if [ ! -f "$STAGE1" ] && [ -f "$SOURCE_ROOT/$STAGE1" ]; then STAGE1="$SOURCE_ROOT/$STAGE1"; fi

[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "--envs must be positive" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
if [ -z "$GRAPHICS" ]; then GRAPHICS=$GPU; fi
[[ "$GRAPHICS" =~ ^[0-9]+$ ]] || { echo "--graphics must be a non-negative integer" >&2; exit 2; }

MASTEER="$SOURCE_ROOT/TokenHSI-masteer"
COORD="$SOURCE_ROOT/TokenHSI-coord"
[ -f "$MASTEER/tokenhsi/run_e2e_stack.py" ] || {
    echo "e2e source checkout missing run_e2e_stack.py: $SOURCE_ROOT" >&2
    exit 2
}
[ -d "$COORD/e2e_planner" ] || {
    echo "e2e source checkout missing e2e_planner: $SOURCE_ROOT" >&2
    exit 2
}
for file in "$PLANNER" "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint not found: $file" >&2; exit 2; }
done
PLANNER=$(realpath -- "$PLANNER")
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")

CONDA_BASE_VALUE=${CONDA_BASE:-/home/injesus1010/anaconda3}
ENV_NAME=${TOKENHSI_CONDA_ENV:-tokenhsi118}
PYTHON_BIN="$CONDA_BASE_VALUE/envs/$ENV_NAME/bin/python"
[ -x "$PYTHON_BIN" ] || {
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
}
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$CONDA_BASE_VALUE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_BASE_VALUE/envs/$ENV_NAME/bin:$PATH"

PYTHONPATH="$COORD" "$PYTHON_BIN" - "$PLANNER" <<'PY'
import sys
from e2e_planner.checkpoint import load_e2e_checkpoint

try:
    model, payload = load_e2e_checkpoint(sys.argv[1], "cpu")
except Exception as error:
    raise SystemExit(f"incompatible e2e planner checkpoint: {error}")
print("e2e planner checkpoint OK: schema={} step={} history_frames={}".format(
    payload["schema_version"], payload.get("step", 0), model.config.history_frames,
))
PY

echo "source:   $SOURCE_ROOT"
echo "planner:  $PLANNER"
echo "policy:   $POLICY"
echo "setting:  E2E_CARRY_END_OFFSET=${E2E_CARRY_END_OFFSET:-0.15}"
echo "devices:  CUDA $GPU, Vulkan $GRAPHICS"
if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 0
fi

CFG=$(mktemp /tmp/e2e_planner_view.XXXXXX.yaml)
cleanup() { rm -f -- "$CFG"; }
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    -e 's/^  enableDebugVis:.*/  enableDebugVis: True/' \
    "$MASTEER/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

export E2E_SOURCE_ROOT=$SOURCE_ROOT
export DISPLAY=${DISPLAY:-:0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export E2E_CONTROLLER=planner
export E2E_CKPT=$PLANNER
export E2E_ROLE_RULE=${E2E_ROLE_RULE:-cost}
export E2E_RESET_COMMIT=${E2E_RESET_COMMIT:-combined}
export E2E_REPLAN_STEPS=${E2E_REPLAN_STEPS:-6}
export E2E_CONTINUITY_WINDOW=${E2E_CONTINUITY_WINDOW:-1.0}
export E2E_CONTINUITY_TOL=${E2E_CONTINUITY_TOL:-0.3}
export E2E_MODE_ON=${E2E_MODE_ON:-0.5}
export E2E_MODE_GATE=${E2E_MODE_GATE:-off}
export E2E_MODE_LATCH=${E2E_MODE_LATCH:-0}
export E2E_INSTALL_ON_CHANGE=${E2E_INSTALL_ON_CHANGE:-1}
export E2E_CHANGE_ROUTE_M=${E2E_CHANGE_ROUTE_M:-0.5}
export E2E_CHANGE_GOAL_M=${E2E_CHANGE_GOAL_M:-0.02}
export E2E_CHANGE_END_M=${E2E_CHANGE_END_M:-0.10}
export E2E_CHANGE_AHEAD=${E2E_CHANGE_AHEAD:-0}
export E2E_CARRY_END_OFFSET=${E2E_CARRY_END_OFFSET:-0.15}
export E2E_SELFCHECK=0
unset E2E_LOG_DIR E2E_DIAG_LOG

export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1
export MS_ZERO=0
export MS_REWARD_OUTER=1
export MS_POS_C=${MS_POS_C:-1.2}
export MS_VEL_W=1
export MS_SCEN=${MS_SCEN:-free}
export MS_CAM=${MS_CAM:-top}
# Continuous planner speeds rarely form long equal-valued runs. Draw the full
# route ribbon by default; set MS_DRAW_SPEED=1 only for quantized speed bands.
export MS_DRAW_SPEED=${MS_DRAW_SPEED:-0}
export STACK_TASK_MODE=stack
export STACK_ALLOW_HAND_CONTACT=1
export STACK_CARRY_REHEARSAL_PROB=0
export STACK_END_ON_A2_RESUME=0
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-900}
export STACK_BOTTOM_Z_TOL=${STACK_BOTTOM_Z_TOL:-0.05}
export STACK_BOTTOM_DISPLACE_TOL=${STACK_BOTTOM_DISPLACE_TOL:-0.50}
export STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.15}
export STACK_EVAL_BOX_GRID=0

# Provide the unversioned libcuda name expected by this Isaac Gym PhysX build.
ROOT=$TARGET_ROOT
. "$TARGET_ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"

echo "[e2e-view] pink/orange routes; cyan/green steering windows; yellow aim points"
cd "$MASTEER"
exec "$PYTHON_BIN" -u "$TARGET_ROOT/e2e_viewer/run_view.py" \
    --task HumanoidMAE2EStackView \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id "$GRAPHICS" \
    --physx --pipeline gpu \
    --cfg_train "$MASTEER/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    --cfg_env "$CFG" \
    --motion_file "$MASTEER/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs "$ENVS" --seed "$SEED" --test --eval_task carry
