#!/bin/bash
# Run a command on a temporary Xvfb display and expose it through noVNC.
#
#   sh tokenhsi/scripts/multi_agent/run-gui.sh \
#     sh tokenhsi/scripts/multi_agent/ma_carry_test.sh <checkpoint> 2 1 3
#
# Every external dependency is auto-detected on PATH and in the usual system /
# user-local install locations. Override any of them if yours live elsewhere:
#
#   PORT                web (noVNC) port                     default 6080
#   RESOLUTION          virtual screen size                  default 1600x900x24
#   VNC_DIR             prefix of a user-local x11vnc install
#                       (e.g. an unpacked .deb: $VNC_DIR/usr/bin/x11vnc)
#   X11VNC              full path to the x11vnc binary
#   NOVNC_DIR           directory containing vnc.html
#   WEBSOCKIFY          full path to the websockify executable
#   TOKENHSI_CONDA_ENV  conda env to search for websockify   default "tokenhsi"
#   TOKENHSI_GPU        value for CUDA_VISIBLE_DEVICES       default 0

set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WEB_PORT=${PORT:-6080}
RESOLUTION=${RESOLUTION:-1600x900x24}
VNC_DIR=${VNC_DIR:-}
CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}

if [ $# -eq 0 ]; then
    echo "usage: run-gui.sh <command> [args...]" >&2
    exit 1
fi

first_executable() {
    for _candidate in "$@"; do
        [ -n "$_candidate" ] && [ -x "$_candidate" ] && { echo "$_candidate"; return 0; }
    done
    return 1
}

# x11vnc: PATH first, then a user-local install under $VNC_DIR.
if [ -z "${X11VNC:-}" ]; then
    X11VNC=$(command -v x11vnc || true)
fi
if [ -z "$X11VNC" ] && [ -n "$VNC_DIR" ]; then
    X11VNC=$(first_executable "$VNC_DIR/usr/bin/x11vnc" "$VNC_DIR/bin/x11vnc" || true)
fi

# noVNC: wherever vnc.html lives.
if [ -z "${NOVNC_DIR:-}" ]; then
    for candidate in /usr/share/novnc "$HOME/opt/novnc" "$HOME/opt/noVNC" \
                     "$HOME/.local/share/novnc" "$ROOT/novnc"; do
        if [ -f "$candidate/vnc.html" ]; then
            NOVNC_DIR=$candidate
            break
        fi
    done
fi

# websockify: PATH, the conda env, then the copy bundled with noVNC.
if [ -z "${WEBSOCKIFY:-}" ]; then
    WEBSOCKIFY=$(command -v websockify || true)
fi
if [ -z "$WEBSOCKIFY" ]; then
    conda_base=$(conda info --base 2>/dev/null || true)
    WEBSOCKIFY=$(first_executable \
        "${CONDA_PREFIX:-}/bin/websockify" \
        "${conda_base:+$conda_base/envs/$CONDA_ENV/bin/websockify}" \
        "${NOVNC_DIR:-}/utils/websockify/run" || true)
fi

command -v Xvfb >/dev/null || { echo "Xvfb not found on PATH" >&2; exit 1; }
[ -n "$X11VNC" ] || { echo "x11vnc not found; set X11VNC=/path/to/x11vnc or VNC_DIR" >&2; exit 1; }
[ -n "$WEBSOCKIFY" ] || { echo "websockify not found; set WEBSOCKIFY=/path/to/websockify" >&2; exit 1; }
[ -n "${NOVNC_DIR:-}" ] && [ -f "$NOVNC_DIR/vnc.html" ] || \
    { echo "noVNC not found; set NOVNC_DIR to the directory holding vnc.html" >&2; exit 1; }

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

# A user-local x11vnc (unpacked .deb) needs its own libs on the search path;
# a system-installed one does not.
X11VNC_LIBS=""
if [ -n "$VNC_DIR" ]; then
    for libdir in "$VNC_DIR/usr/lib/$(uname -m)-linux-gnu" "$VNC_DIR/usr/lib" "$VNC_DIR/lib"; do
        if [ -d "$libdir" ]; then
            X11VNC_LIBS="$libdir${X11VNC_LIBS:+:$X11VNC_LIBS}"
        fi
    done
fi

LD_LIBRARY_PATH="${X11VNC_LIBS}${X11VNC_LIBS:+:}${LD_LIBRARY_PATH:-}" "$X11VNC" \
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
echo " GPU      physical ${TOKENHSI_GPU:-0} (set TOKENHSI_GPU to change)"
echo " 종료     Ctrl+C"
echo "=============================================================="

cd "$ROOT"
DISPLAY="$DISPLAY_NAME" HEADLESS=0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${TOKENHSI_GPU:-0}" \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=0! \
"$@"
