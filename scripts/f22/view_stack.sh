#!/bin/bash
# A=2 rule-coordinated box-stack viewer with a frozen f22 carry+steer actor.
# The positional checkpoint is always the 289-D low-level f22 base.
# Set STACK_CKPT to restore a trained 319-D stack policy/residual.
#
#   STACK_CKPT=/abs/path/stack/Humanoid_00003000.pth \
#     PORT=6101 MA_GPU=7 bash scripts/f22/view_stack.sh /abs/path/f22/Humanoid_00016000.pth
set -e

SCRIPT_PATH=$(readlink -f "$0")
HERE=$(dirname "$SCRIPT_PATH")
BASE=$(readlink -f "$HERE/../..")

# Support both layouts:
#   <project>/TokenHSI-masteer/scripts/f22/view_stack.sh
#   <project>/scripts/f22/view_stack.sh
if [ -f "$BASE/tokenhsi/run.py" ]; then
    REPO="$BASE"
    PROJECT_ROOT=$(dirname "$REPO")
elif [ -f "$BASE/TokenHSI-masteer/tokenhsi/run.py" ]; then
    PROJECT_ROOT="$BASE"
    REPO="$PROJECT_ROOT/TokenHSI-masteer"
else
    echo "cannot find TokenHSI-masteer from script path: $SCRIPT_PATH"
    exit 1
fi

CKPT=${1:?usage: view_stack.sh <f22-checkpoint> [num-envs]}
ENVS=${2:-1}
STACK_CKPT=${STACK_CKPT:-}
PORT=${PORT:-6101}
MA_GPU=${MA_GPU:-7}
VNC_DIR=${VNC_DIR:-/home/hwanhee/opt/vnc}
NOVNC_DIR=${NOVNC_DIR:-/home/hwanhee/opt/novnc}

[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
RESTORE=()
PARITY_CHECK=1
ACTOR_ONLY=1
if [ -n "$STACK_CKPT" ]; then
    [ -f "$STACK_CKPT" ] || { echo "no such stack checkpoint: $STACK_CKPT"; exit 1; }
    RESTORE=(--checkpoint "$STACK_CKPT")
    PARITY_CHECK=0
    ACTOR_ONLY=0
fi

# Fail before Isaac Gym starts if the base and trained-stack checkpoints were
# swapped. Their observation RMS dimensions are part of the checkpoint contract.
python - "$CKPT" "$STACK_CKPT" <<'PY'
import sys
import torch


def rms_dim(path):
    checkpoint = torch.load(path, map_location="cpu")
    try:
        mean = checkpoint["running_mean_std"]["running_mean"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            "checkpoint has no running_mean_std.running_mean: {}".format(path)
        ) from exc
    return int(mean.numel())


base_path, stack_path = sys.argv[1:]
base_dim = rms_dim(base_path)
if base_dim != 289:
    raise SystemExit(
        "invalid f22 base checkpoint (first argument): expected RMS 289, got {}\n"
        "The first argument must be the original f22 checkpoint; pass the trained "
        "stack checkpoint through STACK_CKPT instead.\n"
        "received: {}".format(base_dim, base_path)
    )

if stack_path:
    stack_dim = rms_dim(stack_path)
    if stack_dim != 319:
        raise SystemExit(
            "invalid STACK_CKPT: expected RMS 319, got {}\nreceived: {}".format(
                stack_dim, stack_path
            )
        )
    print("checkpoint contract: f22 RMS=289, stack RMS=319")
else:
    print("checkpoint contract: f22 RMS=289, no trained stack checkpoint")
PY

[ "$MA_GPU" = "7" ] || {
    echo "view_stack.sh is restricted to physical GPU 7; got MA_GPU=$MA_GPU"
    exit 1
}

DISPNUM=$((PORT - 6059))
DISP=":$DISPNUM"
VNC_PORT=$((5900 + DISPNUM))

for p in "$PORT" "$VNC_PORT"; do
    if ss -tln 2>/dev/null | grep -q ":$p "; then
        echo "port $p is in use; choose another PORT"
        exit 1
    fi
done
[ ! -e "/tmp/.X11-unix/X$DISPNUM" ] || {
    echo "display $DISP is in use; choose another PORT"
    exit 1
}

cleanup() {
    kill ${XVFB_PID:-} ${VNC_PID:-} ${WEB_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

VNC_LOG="/tmp/view_stack_${PORT}_x11vnc.log"
WEB_LOG="/tmp/view_stack_${PORT}_websockify.log"
XVFB_LOG="/tmp/view_stack_${PORT}_xvfb.log"

wait_for_listener() {
    local pid=$1 port=$2 name=$3 log=$4
    local i
    for i in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name failed to start; log: $log"
            tail -n 100 "$log" 2>/dev/null || true
            exit 1
        fi
        if ss -tln 2>/dev/null | grep -q ":$port "; then
            return
        fi
        sleep 0.1
    done
    echo "$name did not listen on port $port; log: $log"
    tail -n 100 "$log" 2>/dev/null || true
    exit 1
}

Xvfb "$DISP" -screen 0 1600x900x24 -nolisten tcp >"$XVFB_LOG" 2>&1 &
XVFB_PID=$!
for _ in $(seq 1 30); do
    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
        echo "Xvfb failed to start on $DISP; log: $XVFB_LOG"
        tail -n 100 "$XVFB_LOG" 2>/dev/null || true
        exit 1
    fi
    [ -S "/tmp/.X11-unix/X$DISPNUM" ] && break
    sleep 0.1
done
[ -S "/tmp/.X11-unix/X$DISPNUM" ] || {
    echo "Xvfb socket did not appear for $DISP; log: $XVFB_LOG"
    tail -n 100 "$XVFB_LOG" 2>/dev/null || true
    exit 1
}
LD_LIBRARY_PATH="$VNC_DIR/usr/lib/x86_64-linux-gnu" \
    "$VNC_DIR/usr/bin/x11vnc" -display "$DISP" -rfbport "$VNC_PORT" \
    -localhost -nopw -forever -shared -noxdamage -noshm >"$VNC_LOG" 2>&1 &
VNC_PID=$!
wait_for_listener "$VNC_PID" "$VNC_PORT" x11vnc "$VNC_LOG"
websockify --web="$NOVNC_DIR" "127.0.0.1:$PORT" "127.0.0.1:$VNC_PORT" \
    >"$WEB_LOG" 2>&1 &
WEB_PID=$!
wait_for_listener "$WEB_PID" "$PORT" websockify "$WEB_LOG"

echo "=============================================================="
echo " open     http://localhost:$PORT/vnc.html"
echo " forward  ssh -L $PORT:localhost:$PORT $(hostname)"
echo " f22 base $CKPT"
echo " stack ckpt ${STACK_CKPT:-<none: zero residual>}"
echo " envs     $ENVS x 2 agents (stack FSM, zero coord residual)"
echo " logs     $XVFB_LOG $VNC_LOG $WEB_LOG"
echo "=============================================================="

cd "$REPO"
export DISPLAY="$DISP"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Viewer needs CUDA/Vulkan interop.  CUDA_VISIBLE_DEVICES=7 would turn the
# compute id into logical 0 while BaseTask also passes that 0 to the Vulkan
# graphics device, which selects physical GPU 0 on this server.  Keep physical
# ordinals visible and pass 7 explicitly to compute, RL and graphics instead.
unset CUDA_VISIBLE_DEVICES
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
    echo "warning: PhysX CUDA library directory missing: $PHYSX_LIB_DIR"
fi

echo "CUDA/PhysX cuda:$MA_GPU; RL cuda:$MA_GPU; Vulkan graphics physical GPU $MA_GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="$MA_GPU" --query-gpu=index,name,uuid --format=csv,noheader
fi

# f22_long_c20 contract. Callers can override individual values explicitly.
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
F22_ACTOR_ONLY=$ACTOR_ONLY MS_SCEN=${MS_SCEN:-free} MS_DBG=${MS_DBG:-0} \
STACK_PARITY_CHECK=$PARITY_CHECK STACK_CURRICULUM=0 \
MA_TOKEN=zero MA_C=0 MA_BETA=0 \
python ./tokenhsi/run.py --task HumanoidMAF22StackCarry \
    --sim_device cuda:7 --rl_device cuda:7 --graphics_device_id 7 \
    --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_f22.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_ma_f22_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$CKPT" \
    "${RESTORE[@]}" \
    --test --num_envs "$ENVS"
