#!/bin/bash
# Run any TokenHSI command with an interactive viewer you can reach from a browser.
#
#   bash tokenhsi_gui.sh                                      # default: stage1 test, port 6100
#   bash tokenhsi_gui.sh --port 7000 sh tokenhsi/scripts/single_task/sit_test.sh
#   bash tokenhsi_gui.sh --port 7001 --res 1920x1080x24 python ./tokenhsi/run.py --task HumanoidTraj ...
#
# Options must come before the command:
#   --port N     browser port (default 6100)
#   --gpu N      physical GPU 6 or 7 (default 6)
#   --display :N X display to use (default: first free from :99)
#   --res WxHxD  virtual screen size (default 1600x900x24)
#
# Then on your laptop:  ssh -L <port>:localhost:<port> <this-host>
# and open http://localhost:<port>/vnc.html
set -e

WEB_PORT=6100
DISP=""
RES=1600x900x24
GPU=${GUI_GPU:-6}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VNC_DIR=/home/hwanhee/opt/vnc
NOVNC_DIR=/home/hwanhee/opt/novnc

while [[ "$1" == --* ]]; do
    case "$1" in
        --port)    WEB_PORT="$2"; shift 2 ;;
        --gpu)     GPU="$2";      shift 2 ;;
        --display) DISP="$2";     shift 2 ;;
        --res)     RES="$2";      shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

VIEW_DRI=$(python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$GPU")

# pick a free display if not given
if [ -z "$DISP" ]; then
    for n in $(seq 99 130); do
        if [ ! -e /tmp/.X11-unix/X$n ]; then DISP=":$n"; break; fi
    done
fi
DISP_NUM=${DISP#:}
VNC_PORT=$((5900 + DISP_NUM))

if ss -tln 2>/dev/null | grep -q ":$WEB_PORT "; then
    echo "port $WEB_PORT is already in use — pick another with --port"; exit 1
fi

REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d "$ROOT/$REPO/tokenhsi" ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd "$ROOT/$REPO"

cleanup() { kill $XVFB_PID $VNC_PID $WEB_PID 2>/dev/null || true; }
trap cleanup EXIT INT TERM

Xvfb $DISP -screen 0 $RES -nolisten tcp &
XVFB_PID=$!
sleep 2

LD_LIBRARY_PATH=$VNC_DIR/usr/lib/x86_64-linux-gnu $VNC_DIR/usr/bin/x11vnc \
    -display $DISP -rfbport $VNC_PORT -localhost -nopw -forever -shared -noxdamage -noshm -quiet &
VNC_PID=$!
sleep 1

websockify --web=$NOVNC_DIR 127.0.0.1:$WEB_PORT 127.0.0.1:$VNC_PORT > /dev/null 2>&1 &
WEB_PID=$!
sleep 1

echo "=============================================================="
echo " display $DISP   vnc :$VNC_PORT   ->  http://localhost:$WEB_PORT/vnc.html"
echo " from your laptop: ssh -L $WEB_PORT:localhost:$WEB_PORT $(hostname)"
echo " GPU physical $GPU (CUDA logical 0, Vulkan $VIEW_DRI)"
echo " mouse: right-drag + WASD to fly   F: follow   K: lines   L: record"
echo "=============================================================="

if [ $# -eq 0 ]; then
    set -- sh tokenhsi/scripts/tokenhsi/stage1_test.sh
fi

DISPLAY=$DISP CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=$VIEW_DRI "$@"
