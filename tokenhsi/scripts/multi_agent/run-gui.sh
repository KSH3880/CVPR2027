#!/bin/bash
# Run a command on a temporary Xvfb display and expose it through noVNC.
#
#   sh tokenhsi/scripts/multi_agent/run-gui.sh \
#     sh tokenhsi/scripts/multi_agent/ma_carry_test.sh <checkpoint> 2 1 3

set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WEB_PORT=${PORT:-6080}
RESOLUTION=${RESOLUTION:-1600x900x24}
VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
NOVNC_DIR=${NOVNC_DIR:-$HOME/opt/novnc}

if [ $# -eq 0 ]; then
    echo "usage: run-gui.sh <command> [args...]" >&2
    exit 1
fi

X11VNC="$VNC_DIR/usr/bin/x11vnc"
WEBSOCKIFY=$(command -v websockify || true)
if [ -z "$WEBSOCKIFY" ]; then
    conda_base=$(conda info --base 2>/dev/null || true)
    candidate="$conda_base/envs/tokenhsi_jhh/bin/websockify"
    [ -x "$candidate" ] && WEBSOCKIFY=$candidate
fi

command -v Xvfb >/dev/null || { echo "Xvfb not found" >&2; exit 1; }
[ -x "$X11VNC" ] || { echo "x11vnc not found: $X11VNC" >&2; exit 1; }
[ -n "$WEBSOCKIFY" ] || { echo "websockify not found in tokenhsi_jhh" >&2; exit 1; }
[ -f "$NOVNC_DIR/vnc.html" ] || { echo "noVNC not found: $NOVNC_DIR" >&2; exit 1; }

if ss -ltn 2>/dev/null | grep -q ":$WEB_PORT "; then
    echo "port $WEB_PORT is already in use; choose another PORT" >&2
    exit 1
fi

DISPLAY_NUM=""
for n in $(seq 99 130); do
    if [ ! -e "/tmp/.X11-unix/X$n" ] && [ ! -e "/tmp/.X$n-lock" ]; then
        DISPLAY_NUM=$n
        break
    fi
done
[ -n "$DISPLAY_NUM" ] || { echo "no free X display found" >&2; exit 1; }

DISPLAY_NAME=:$DISPLAY_NUM
VNC_PORT=$((5900 + DISPLAY_NUM))
if ss -ltn 2>/dev/null | grep -q ":$VNC_PORT "; then
    echo "VNC port $VNC_PORT is already in use" >&2
    exit 1
fi

XVFB_LOG=/tmp/tokenhsi_gui_${WEB_PORT}_xvfb.log
VNC_LOG=/tmp/tokenhsi_gui_${WEB_PORT}_x11vnc.log
WEB_LOG=/tmp/tokenhsi_gui_${WEB_PORT}_websockify.log

cleanup() {
    kill ${XVFB_PID:-} ${VNC_PID:-} ${WEB_PID:-} 2>/dev/null || true
    wait ${XVFB_PID:-} ${VNC_PID:-} ${WEB_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() {
    pid=$1
    port=$2
    name=$3
    log=$4
    for _ in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name failed; log: $log" >&2
            return 1
        fi
        if ss -ltn 2>/dev/null | grep -q ":$port "; then
            return 0
        fi
        sleep 0.1
    done
    echo "$name did not open port $port; log: $log" >&2
    return 1
}

Xvfb "$DISPLAY_NAME" -screen 0 "$RESOLUTION" -nolisten tcp >"$XVFB_LOG" 2>&1 &
XVFB_PID=$!
for _ in $(seq 1 30); do
    [ -e "/tmp/.X11-unix/X$DISPLAY_NUM" ] && break
    kill -0 "$XVFB_PID" 2>/dev/null || { echo "Xvfb failed; log: $XVFB_LOG" >&2; exit 1; }
    sleep 0.1
done
[ -e "/tmp/.X11-unix/X$DISPLAY_NUM" ] || { echo "Xvfb did not start; log: $XVFB_LOG" >&2; exit 1; }

LD_LIBRARY_PATH="$VNC_DIR/usr/lib/x86_64-linux-gnu" "$X11VNC" \
    -display "$DISPLAY_NAME" -rfbport "$VNC_PORT" -localhost -nopw -forever \
    -shared -noxdamage -noshm -quiet >"$VNC_LOG" 2>&1 &
VNC_PID=$!
wait_for_port "$VNC_PID" "$VNC_PORT" x11vnc "$VNC_LOG"

"$WEBSOCKIFY" --web="$NOVNC_DIR" "127.0.0.1:$WEB_PORT" \
    "127.0.0.1:$VNC_PORT" >"$WEB_LOG" 2>&1 &
WEB_PID=$!
wait_for_port "$WEB_PID" "$WEB_PORT" websockify "$WEB_LOG"

echo "=============================================================="
echo " noVNC    http://localhost:$WEB_PORT/vnc.html?autoconnect=1&resize=remote"
echo " VS Code  PORTS에서 $WEB_PORT 포워딩"
echo " display  $DISPLAY_NAME"
echo " GPU      physical 0 (CUDA logical 0)"
echo " 종료     Ctrl+C"
echo "=============================================================="

cd "$ROOT"
DISPLAY="$DISPLAY_NAME" HEADLESS=0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=0! \
"$@"
