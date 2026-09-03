#!/bin/bash
# Train only the coordination token/action residual on a frozen f22 actor.
#
# Smoke (always do this first):
#   STACK_ITERS=2 MS_GRADCHK=1 bash scripts/f22/train_stack.sh stack_grad_s0
#
# Pilot:
#   STACK_ITERS=3000 bash scripts/f22/train_stack.sh stack_pilot_s0
#
# Resume into a new output tag (STACK_ITERS is the final iteration):
#   STACK_RESUME_CKPT=/abs/path/Humanoid_00000600.pth \
#     STACK_ITERS=3000 bash scripts/f22/train_stack.sh stack_resume_s0
set -e

SCRIPT_PATH=$(readlink -f "$0")
HERE=$(dirname "$SCRIPT_PATH")
BASE=$(readlink -f "$HERE/../..")

if [ -f "$BASE/tokenhsi/run.py" ]; then
    REPO="$BASE"
elif [ -f "$BASE/TokenHSI-masteer/tokenhsi/run.py" ]; then
    REPO="$BASE/TokenHSI-masteer"
else
    echo "cannot find TokenHSI-masteer from script path: $SCRIPT_PATH"
    exit 1
fi
PROJECT_ROOT=$(dirname "$REPO")

TAG=${1:?usage: train_stack.sh <tag> [f22-checkpoint]}
DEFAULT_F22_CKPT="/home/hwanhee/CVPR2027/TokenHSI-steer/output/steer/f22b_long_c20/Humanoid_19-22-50-01/nn/Humanoid_00016000.pth"
CKPT=${2:-"${F22_CKPT:-$DEFAULT_F22_CKPT}"}
ENVS=${STACK_ENVS:-2048}
ITERS=${STACK_ITERS:-2}
SEED=${STACK_SEED:-0}
GPU=${MA_GPU:-7}
OUT="output/stack/$TAG"
SAVE_FREQ=${STACK_SAVE_FREQ:-100}
CURRICULUM=${STACK_CURRICULUM:-1}
TASK_MODE=${STACK_TASK_MODE:-stack}
CURR_STAGE1_ITERS=${STACK_CURR_STAGE1_ITERS:-750}
CURR_STAGE2_ITERS=${STACK_CURR_STAGE2_ITERS:-1800}
HORIZON=32
DEFAULT_MINIBATCH=16384
RESUME_CKPT=${STACK_RESUME_CKPT:-}
RESTORE=()

# rl_games flattens both agents into the PPO batch:
#   batch_size = num_envs * horizon_length * num_agents.
# Keep the production default when it divides the batch, and otherwise choose
# the largest power-of-two divisor so small smoke runs remain valid.
BATCH_SIZE=$((ENVS * HORIZON * 2))
if [ -n "${STACK_MINIBATCH_SIZE:-}" ]; then
    MINIBATCH_SIZE=$STACK_MINIBATCH_SIZE
else
    MINIBATCH_SIZE=$DEFAULT_MINIBATCH
    if [ "$BATCH_SIZE" -lt "$MINIBATCH_SIZE" ] || [ $((BATCH_SIZE % MINIBATCH_SIZE)) -ne 0 ]; then
        MINIBATCH_SIZE=1
        while [ $((MINIBATCH_SIZE * 2)) -le "$DEFAULT_MINIBATCH" ] \
            && [ $((MINIBATCH_SIZE * 2)) -le "$BATCH_SIZE" ] \
            && [ $((BATCH_SIZE % (MINIBATCH_SIZE * 2))) -eq 0 ]; do
            MINIBATCH_SIZE=$((MINIBATCH_SIZE * 2))
        done
    fi
fi

[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
if [ -n "$RESUME_CKPT" ]; then
    [ -f "$RESUME_CKPT" ] || { echo "no such resume checkpoint: $RESUME_CKPT"; exit 1; }
    RESTORE=(--checkpoint "$RESUME_CKPT")
fi
case "$GPU" in
    0|1|2|3|4|5|6|7) ;;
    *) echo "MA_GPU must be a physical GPU index in 0..7; got $GPU"; exit 1 ;;
esac
case "$TAG" in
    ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG"; exit 1 ;;
esac
[ ! -e "$REPO/$OUT" ] || {
    echo "refusing to reuse existing output tag: $REPO/$OUT"
    exit 1
}
case "$ENVS:$ITERS:$SAVE_FREQ:$CURR_STAGE1_ITERS:$CURR_STAGE2_ITERS:$MINIBATCH_SIZE" in
    *[!0-9:]*|:*|*:) echo "stack counts must be positive integers"; exit 1 ;;
esac
[ "$ENVS" -gt 0 ] && [ "$ITERS" -gt 0 ] && [ "$SAVE_FREQ" -gt 0 ] || {
    echo "STACK_ENVS, STACK_ITERS and STACK_SAVE_FREQ must be positive"
    exit 1
}
[ "$MINIBATCH_SIZE" -gt 0 ] && [ "$MINIBATCH_SIZE" -le "$BATCH_SIZE" ] \
    && [ $((BATCH_SIZE % MINIBATCH_SIZE)) -eq 0 ] || {
    echo "STACK_MINIBATCH_SIZE must divide PPO batch $BATCH_SIZE; got $MINIBATCH_SIZE"
    exit 1
}
case "$CURRICULUM" in
    0|1) ;;
    *) echo "STACK_CURRICULUM must be 0 or 1"; exit 1 ;;
esac
case "$TASK_MODE" in
    stack|putdown_retreat) ;;
    *) echo "STACK_TASK_MODE must be stack or putdown_retreat"; exit 1 ;;
esac
[ "$CURR_STAGE1_ITERS" -gt 0 ] && [ "$CURR_STAGE2_ITERS" -gt "$CURR_STAGE1_ITERS" ] || {
    echo "curriculum boundaries must satisfy 0 < stage1 < stage2"
    exit 1
}
# rl_games frame counts environment transitions (horizon * num_envs), not
# flattened agent rows, so the two-agent factor is intentionally absent.
CURR_STAGE1_FRAMES=$((CURR_STAGE1_ITERS * ENVS * HORIZON))
CURR_STAGE2_FRAMES=$((CURR_STAGE2_ITERS * ENVS * HORIZON))

cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
unset TOKENHSI_GRAPHICS_DEVICE_ID
export MS_GRADCHK=${MS_GRADCHK:-1}
PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
    echo "warning: PhysX CUDA library directory missing: $PHYSX_LIB_DIR"
fi

if [ -n "$RESUME_CKPT" ]; then
    python - "$RESUME_CKPT" "$ITERS" <<'PY'
import sys
import torch

path, final_iter = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(path, map_location="cpu")
try:
    rms_dim = int(checkpoint["running_mean_std"]["running_mean"].numel())
except (KeyError, TypeError) as exc:
    raise SystemExit(
        "resume checkpoint has no running_mean_std.running_mean: {}".format(path)
    ) from exc
if rms_dim != 319:
    raise SystemExit(
        "resume checkpoint must be a 319-D stack policy, got RMS {}: {}".format(
            rms_dim, path
        )
    )
epoch = int(checkpoint.get("epoch", 0))
frame = int(checkpoint.get("frame", 0))
if final_iter <= epoch:
    raise SystemExit(
        "STACK_ITERS is the final iteration and must exceed checkpoint epoch "
        "{}; got {}".format(epoch, final_iter)
    )
print("resume contract: stack RMS=319 epoch={} frame={} -> final_iter={}".format(
    epoch, frame, final_iter
))
PY
fi

# Preserve numbered checkpoints instead of only overwriting Humanoid.pth.
TRAIN_CFG_DIR="$PROJECT_ROOT/runs/gen_cfgs/stack"
TRAIN_CFG="$TRAIN_CFG_DIR/${TAG}_train.yaml"
mkdir -p "$TRAIN_CFG_DIR"
sed -e "s/^\( *\)save_frequency:.*/\1save_frequency: $SAVE_FREQ/" \
    -e "s/^\( *\)save_intermediate:.*/\1save_intermediate: True/" \
    -e "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MINIBATCH_SIZE/" \
    tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_f22.yaml \
    > "$TRAIN_CFG"

echo "=============================================================="
echo " task       HumanoidMAF22StackCarry"
echo " task mode  $TASK_MODE"
echo " f22 base   $CKPT"
echo " resume     ${RESUME_CKPT:-<none: new residual>}"
echo " output     $OUT"
echo " envs       $ENVS x 2 agents"
echo " PPO batch  $BATCH_SIZE (minibatch $MINIBATCH_SIZE)"
echo " iterations $ITERS"
echo " seed       $SEED"
echo " GPU        physical $GPU -> logical cuda:0"
echo " grad check $MS_GRADCHK"
echo " coord w    ${STACK_COORD_REWARD_W:-0.25}"
echo " retreat    native carry reward x ${STACK_RETREAT_BASE_REWARD_W:-0.0}"
echo " retreat    time penalty ${STACK_RETREAT_TIME_PENALTY:-0.1}"
echo " wait       A2 native carry reward x ${STACK_WAIT_BASE_REWARD_W:-0.0}"
echo " collision body=${STACK_BODY_CLEARANCE:-0.22} m cross-box=${STACK_CROSS_BOX_CLEARANCE:-0.22} m weight=${STACK_COLLISION_PENALTY:-0.35}"
echo " residual   pickup<=${STACK_PICKUP_RESIDUAL_SCALE:-0.05} full<=${STACK_RESIDUAL_SCALE:-0.15}"
echo " top tol    xy=${STACK_TOP_XY_TOL:-0.08} m z=${STACK_TOP_Z_TOL:-0.08} m"
echo " curriculum phase-task=$CURRICULUM stages=0..$CURR_STAGE1_ITERS..$CURR_STAGE2_ITERS iter"
echo " save every $SAVE_FREQ iterations (numbered checkpoints)"
echo " PhysX CUDA $PHYSX_LIB_DIR"
echo "=============================================================="

# f22b_long_c20 command contract plus stack coordination knobs.
STEER_K=${STEER_K:-6} \
STEER_SPACING=${STEER_SPACING:-0.4} \
STEER_DUAL=${STEER_DUAL:-1} \
STEER_PHASEFREE=${STEER_PHASEFREE:-1} \
STEER_TRAJ=${STEER_TRAJ:-v3} \
STEER_POS=${STEER_POS:-latpen} \
STEER_POS_C=${STEER_POS_C:-2.0} \
STEER_VEL=${STEER_VEL:-aim} \
STEER_LOOKAHEAD=${STEER_LOOKAHEAD:-1.2} \
STEER_CURVATURE=${STEER_CURVATURE:-0.15} \
STEER_CELL=${STEER_CELL:-60} \
MS_SCEN=${MS_SCEN:-free} MS_DBG=${MS_DBG:-0} \
STACK_COORD_REWARD_W=${STACK_COORD_REWARD_W:-0.25} \
STACK_RETREAT_BASE_REWARD_W=${STACK_RETREAT_BASE_REWARD_W:-0.0} \
STACK_RETREAT_TIME_PENALTY=${STACK_RETREAT_TIME_PENALTY:-0.1} \
STACK_WAIT_BASE_REWARD_W=${STACK_WAIT_BASE_REWARD_W:-0.0} \
STACK_BODY_CLEARANCE=${STACK_BODY_CLEARANCE:-0.22} \
STACK_CROSS_BOX_CLEARANCE=${STACK_CROSS_BOX_CLEARANCE:-0.22} \
STACK_COLLISION_PENALTY=${STACK_COLLISION_PENALTY:-0.35} \
STACK_PICKUP_RESIDUAL_SCALE=${STACK_PICKUP_RESIDUAL_SCALE:-0.05} \
STACK_RESIDUAL_SCALE=${STACK_RESIDUAL_SCALE:-0.15} \
STACK_RESIDUAL_DEBUG=${STACK_RESIDUAL_DEBUG:-0} \
STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.08} \
STACK_TOP_Z_TOL=${STACK_TOP_Z_TOL:-0.08} \
STACK_HOLD_RADIUS=${STACK_HOLD_RADIUS:-0.45} \
STACK_STABLE_STEPS=${STACK_STABLE_STEPS:-20} \
STACK_TASK_MODE=$TASK_MODE \
STACK_CURRICULUM=$CURRICULUM \
STACK_CURR_STAGE1_FRAMES=$CURR_STAGE1_FRAMES \
STACK_CURR_STAGE2_FRAMES=$CURR_STAGE2_FRAMES \
MA_C=0 MA_BETA=0 \
python -u ./tokenhsi/run.py --task HumanoidMAF22StackCarry \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 \
    --physx --pipeline gpu --headless \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_ma_f22_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$CKPT" \
    "${RESTORE[@]}" \
    --num_envs "$ENVS" --max_iterations "$ITERS" \
    --output_path "$OUT" --seed "$SEED"
