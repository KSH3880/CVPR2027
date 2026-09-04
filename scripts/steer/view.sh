#!/bin/bash
# Watch a trained steering policy in the IsaacGym viewer, over VNC.
#
#   PORT=6080 bash scripts/steer/view.sh
#   PORT=6081 bash scripts/steer/view.sh <ckpt> <num_envs>
#   PORT=6082 STEER_CURVATURE=0.35 bash scripts/steer/view.sh
#   RECORD=1 PORT=6080 bash scripts/steer/view.sh      # + mp4 into runs/videos/steer/
#   RECORD=1 RECORD_DELAY=75 ...                       # start recording after the sim loads
#
# The X display and VNC port are derived from PORT, so several viewers can run
# at once on different ports without colliding.
#
# Then forward it and open the page:
#   ssh -L <PORT>:localhost:<PORT> <host>
#   http://localhost:<PORT>/vnc.html
#
# Overlay: grey line on the ground is the whole commanded path (ending at the
# drop point), cyan line at root height is the window fed to the policy, amber
# post is the point the reward aims at. STEER_DRAW_TASK=1 restores TokenHSI's
# own task lines. STEER_ZERO=1 turns steering off for comparison.
set -e
# Resolve the copied checkout instead of silently running the original workspace.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VNC_DIR=/home/hwanhee/opt/vnc
NOVNC_DIR=/home/hwanhee/opt/novnc

PORT=${PORT:-6080}
# default: the best F2 run (re-aim + cross-track, whole-task path)
CKPT=${1:-output/steer/f2_3_D/Humanoid_14-14-30-06/nn/Humanoid.pth}
ENVS=${2:-2}
VIEW_GPU=${STEER_GPU:-6}
VIEW_DRI=$(python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$VIEW_GPU")

DISPNUM=$((PORT - 6059))          # 6080 -> :21
DISP=":$DISPNUM"
VNC_PORT=$((5900 + DISPNUM))

cd $ROOT/TokenHSI-steer
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
for p in $PORT $VNC_PORT; do
    if ss -tln 2>/dev/null | grep -q ":$p "; then
        echo "port $p is in use - run with a different PORT, e.g. PORT=$((PORT+1))"; exit 1
    fi
done

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

cleanup() {
    # SIGINT first so ffmpeg writes the mp4 trailer instead of leaving a stub
    if [ -n "$FFMPEG_PID" ]; then
        kill -INT $FFMPEG_PID 2>/dev/null || true
        wait $FFMPEG_PID 2>/dev/null || true
    fi
    kill $XVFB_PID $VNC_PID $WEB_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

Xvfb $DISP -screen 0 1600x900x24 -nolisten tcp &
XVFB_PID=$!
sleep 2
LD_LIBRARY_PATH=$VNC_DIR/usr/lib/x86_64-linux-gnu $VNC_DIR/usr/bin/x11vnc \
    -display $DISP -rfbport $VNC_PORT -localhost -nopw -forever -shared -noxdamage -quiet &
VNC_PID=$!
sleep 1
websockify --web=$NOVNC_DIR 127.0.0.1:$PORT 127.0.0.1:$VNC_PORT > /dev/null 2>&1 &
WEB_PID=$!
sleep 1

if [ "${RECORD:-0}" != 0 ]; then
    TAG=$(echo "$CKPT" | sed -E 's|.*/output/steer/([^/]+)/.*|\1|; s|/|_|g')
    VIDEO=$ROOT/runs/videos/steer/${TAG}_$(date +%Y%m%d-%H%M%S).mp4
    mkdir -p "$(dirname "$VIDEO")"
    # crop off the docked Isaac Gym panel (fixed 323 px on the left) and the
    # pointer; RECORD_DELAY skips the black screen while the sim loads
    (sleep ${RECORD_DELAY:-0}; exec ffmpeg -nostdin -loglevel error -y \
        -f x11grab -draw_mouse 0 -framerate 30 \
        -video_size 1276x900 -i $DISP+324,0 \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$VIDEO") &
    FFMPEG_PID=$!
fi

echo "=============================================================="
echo " open     http://localhost:$PORT/vnc.html"
echo " forward  ssh -L $PORT:localhost:$PORT $(hostname)"
echo " display  $DISP    vnc  $VNC_PORT"
echo " ckpt     $CKPT"
echo " GPU      physical $VIEW_GPU (CUDA logical 0, Vulkan $VIEW_DRI)"
echo " envs     $ENVS    curvature ${STEER_CURVATURE:-0.15}"
if [ -n "$VIDEO" ]; then echo " video    $VIDEO"; fi
echo "=============================================================="

DISPLAY=$DISP \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$VIEW_GPU \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=$VIEW_DRI \
STEER_ZERO=${STEER_ZERO:-0} \
STEER_CELL=${STEER_CELL:-60} \
STEER_PHASEFREE=${STEER_PHASEFREE:-0} \
STEER_CURVATURE=${STEER_CURVATURE:-0.15} \
STEER_REAIM=1 STEER_AIM=lookahead \
STEER_LOOKAHEAD=${STEER_LOOKAHEAD:-1.2} \
STEER_XTRACK=${STEER_XTRACK:-1.0} \
python ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train "$ROOT/runs/gen_cfgs/steer/f2_base.yaml" \
    --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry_cp32.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint "$CKPT" \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
    --test --num_envs "$ENVS"
