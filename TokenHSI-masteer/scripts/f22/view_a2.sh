#!/bin/bash
# A=2 separated-scene baseline viewer for the single-agent f22 carry+steer checkpoint.
# This path intentionally does not use HumanoidMASteerCarry or the modified
# masteer adapt builder.
#
#   PORT=6101 MA_GPU=7 bash scripts/f22/view_a2.sh /abs/path/Humanoid_00016000.pth
set -e

SCRIPT_PATH=$(readlink -f "$0")
HERE=$(dirname "$SCRIPT_PATH")
BASE=$(readlink -f "$HERE/../..")

# Support both layouts:
#   <project>/TokenHSI-masteer/scripts/f22/view_a2.sh
#   <project>/scripts/f22/view_a2.sh
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

CKPT=${1:?usage: view_a2.sh <f22-checkpoint> [num-envs]}
ENVS=${2:-1}
PORT=${PORT:-6101}
MA_GPU=${MA_GPU:-0}
VNC_DIR=${VNC_DIR:-/home/hwanhee/opt/vnc}
NOVNC_DIR=${NOVNC_DIR:-/home/hwanhee/opt/novnc}

[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

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

Xvfb "$DISP" -screen 0 1600x900x24 -nolisten tcp &
XVFB_PID=$!
sleep 2
LD_LIBRARY_PATH="$VNC_DIR/usr/lib/x86_64-linux-gnu" \
    "$VNC_DIR/usr/bin/x11vnc" -display "$DISP" -rfbport "$VNC_PORT" \
    -localhost -nopw -forever -shared -noxdamage -quiet &
VNC_PID=$!
sleep 1
websockify --web="$NOVNC_DIR" "127.0.0.1:$PORT" "127.0.0.1:$VNC_PORT" \
    >/dev/null 2>&1 &
WEB_PID=$!
sleep 1

echo "=============================================================="
echo " open     http://localhost:$PORT/vnc.html"
echo " forward  ssh -L $PORT:localhost:$PORT $(hostname)"
echo " checkpoint $CKPT"
echo " envs     $ENVS x 2 agents (no teammate input)"
echo "=============================================================="

cd "$REPO"
export DISPLAY="$DISP"
export CUDA_VISIBLE_DEVICES="$MA_GPU"
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
    echo "warning: PhysX CUDA library directory missing: $PHYSX_LIB_DIR"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (IsaacGym logical device cuda:0)"

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
F22_ACTOR_ONLY=1 MS_SCEN=${MS_SCEN:-solo} MS_SEP=${MS_SEP:-9} MS_DBG=${MS_DBG:-0} \
MA_TOKEN=zero MA_C=0 MA_BETA=0 \
python ./tokenhsi/run.py --task HumanoidMAF22SteerCarry \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
    --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_f22.yaml \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_ma_f22_steer_carry.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint "$CKPT" \
    --test --num_envs "$ENVS"
