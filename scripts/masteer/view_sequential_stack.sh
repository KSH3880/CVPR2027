#!/bin/bash
# Serve the sequential-stack Isaac Gym viewer through noVNC on loopback.
# The simulation and noVNC services run as the SSH user who starts this script.
# The visitor account can run everything read-only from the shared project.
#
# On the visitor's own PC, start the viewer and SSH tunnel together:
#   ssh -t -L 6100:127.0.0.1:6100 visitor@<this-host> \
#     'MA_GPU=0 PORT=6100 bash /home/injesus1010/repos/CVPR2027/scripts/masteer/view_sequential_stack.sh'
# Then open http://localhost:6100/vnc.html on that PC. Keep SSH connected.
#
# Optional positional arguments are the same as the local viewer:
#   bash scripts/masteer/view_sequential_stack.sh /abs/policy.pth 1
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOCAL_VIEW="$ROOT/scripts/masteer/view_sequential_stack_local.sh"
RUNTIME=${NOVNC_RUNTIME:-"$ROOT/runs/tools/novnc-runtime"}

PORT=${PORT:-6100}
RES=${NOVNC_RES:-1600x900x24}
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 6060 ] || [ "$PORT" -gt 65000 ]; then
    echo "PORT는 6060~65000 정수여야 한다: $PORT" >&2
    exit 2
fi

DISPNUM=$((PORT - 6059))
DISP=":$DISPNUM"
VNC_PORT=$((5900 + DISPNUM))

find_program() {
    local name=$1 bundled=$2 found
    if found=$(command -v "$name" 2>/dev/null); then
        printf '%s\n' "$found"
    elif [ -x "$bundled" ]; then
        printf '%s\n' "$bundled"
    else
        return 1
    fi
}

if ! XVFB=$(find_program Xvfb "$RUNTIME/usr/bin/Xvfb"); then
    echo "Xvfb 없음. NOVNC_RUNTIME 경로를 확인한다: $RUNTIME" >&2
    exit 1
fi
if ! X11VNC=$(find_program x11vnc "$RUNTIME/usr/bin/x11vnc"); then
    echo "x11vnc 없음. NOVNC_RUNTIME 경로를 확인한다: $RUNTIME" >&2
    exit 1
fi
if ! WEBSOCKIFY=$(find_program websockify "$RUNTIME/usr/bin/websockify"); then
    echo "websockify 없음. NOVNC_RUNTIME 경로를 확인한다: $RUNTIME" >&2
    exit 1
fi
NOVNC_WEB=${NOVNC_WEB:-"$RUNTIME/usr/share/novnc"}
if [ ! -f "$NOVNC_WEB/vnc.html" ]; then
    echo "noVNC 웹 파일 없음: $NOVNC_WEB/vnc.html" >&2
    exit 1
fi
if [ ! -x "$LOCAL_VIEW" ]; then
    echo "로컬 viewer 스크립트 없음: $LOCAL_VIEW" >&2
    exit 1
fi

port_is_listening() {
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

for p in "$PORT" "$VNC_PORT"; do
    if port_is_listening "$p"; then
        echo "포트 $p 사용 중. 다른 PORT를 지정한다 (예: PORT=$((PORT + 1)))." >&2
        exit 1
    fi
done
if [ -e "/tmp/.X11-unix/X$DISPNUM" ]; then
    echo "디스플레이 $DISP 사용 중. 다른 PORT를 지정한다." >&2
    exit 1
fi

XVFB_LOG="/tmp/sequential_stack_${PORT}_xvfb.log"
VNC_LOG="/tmp/sequential_stack_${PORT}_x11vnc.log"
WEB_LOG="/tmp/sequential_stack_${PORT}_websockify.log"

cleanup() {
    local pid
    for pid in "${WEB_PID:-}" "${VNC_PID:-}" "${XVFB_PID:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${WEB_PID:-}" "${VNC_PID:-}" "${XVFB_PID:-}"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    # Xvfb normally removes these itself. Remove only this script's exact
    # display artifacts after its process has exited, so the PORT is reusable.
    if [ -n "${XVFB_PID:-}" ] && ! kill -0 "$XVFB_PID" 2>/dev/null; then
        rm -f -- "/tmp/.X11-unix/X$DISPNUM" "/tmp/.X${DISPNUM}-lock"
    fi
}
trap cleanup EXIT INT TERM

wait_for_process() {
    local pid=$1 port=$2 name=$3 log=$4 i
    for i in $(seq 1 50); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name 시작 실패. 로그: $log" >&2
            tail -n 100 "$log" 2>/dev/null || true
            exit 1
        fi
        if port_is_listening "$port"; then
            return 0
        fi
        sleep 0.1
    done
    echo "$name이 포트 $port에서 대기하지 않는다. 로그: $log" >&2
    tail -n 100 "$log" 2>/dev/null || true
    exit 1
}

RUNTIME_LIB="$RUNTIME/usr/lib/x86_64-linux-gnu"
RUNTIME_PY="$RUNTIME/usr/lib/python3/dist-packages"
# x11vnc 0.9.17 refuses to attach to Xvfb when the launching desktop session's
# WAYLAND_DISPLAY leaks into this process, even though -display points to Xvfb.
unset WAYLAND_DISPLAY

"$XVFB" "$DISP" -screen 0 "$RES" -nolisten tcp -ac >"$XVFB_LOG" 2>&1 &
XVFB_PID=$!
for _ in $(seq 1 50); do
    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
        echo "Xvfb 시작 실패. 로그: $XVFB_LOG" >&2
        tail -n 100 "$XVFB_LOG" 2>/dev/null || true
        exit 1
    fi
    [ -S "/tmp/.X11-unix/X$DISPNUM" ] && break
    sleep 0.1
done
if [ ! -S "/tmp/.X11-unix/X$DISPNUM" ]; then
    echo "Xvfb 디스플레이 소켓 생성 실패: $DISP" >&2
    exit 1
fi

LD_LIBRARY_PATH="$RUNTIME_LIB:${LD_LIBRARY_PATH:-}" \
"$X11VNC" -display "$DISP" -rfbport "$VNC_PORT" -localhost -nopw \
    -forever -shared -noxdamage -noshm -quiet >"$VNC_LOG" 2>&1 &
VNC_PID=$!
wait_for_process "$VNC_PID" "$VNC_PORT" x11vnc "$VNC_LOG"

PYTHONPATH="$RUNTIME_PY" \
"$WEBSOCKIFY" --web="$NOVNC_WEB" "127.0.0.1:$PORT" \
    "127.0.0.1:$VNC_PORT" >"$WEB_LOG" 2>&1 &
WEB_PID=$!
wait_for_process "$WEB_PID" "$PORT" websockify "$WEB_LOG"

SSH_HOST=${NOVNC_SSH_HOST:-$(hostname -f 2>/dev/null || hostname)}
RUN_USER=$(id -un)
echo "=============================================================="
echo " noVNC는 이 PC의 loopback에만 열림 (외부 직접 노출 없음)"
echo " 실행 계정:   $RUN_USER"
echo " SSH 연결:    visitor@$SSH_HOST (이 터미널을 계속 유지)"
echo " browser:    http://localhost:$PORT/vnc.html"
echo " display:    $DISP   VNC: 127.0.0.1:$VNC_PORT"
echo " logs:       $XVFB_LOG $VNC_LOG $WEB_LOG"
echo " 종료:       이 터미널에서 Ctrl-C"
echo "=============================================================="

if [ "${NOVNC_SMOKE_ONLY:-0}" = "1" ]; then
    curl --fail --silent --show-error --output /dev/null \
        "http://127.0.0.1:$PORT/vnc.html"
    echo "noVNC smoke OK: Xvfb, VNC, WebSocket/HTTP"
    exit 0
fi

# Xvfb has no hardware GLX display. These match the established remote viewer
# setup while Isaac Gym/PhysX computation still uses MA_GPU through the local
# viewer script.
export DISPLAY="$DISP"
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export MESA_LOADER_DRIVER_OVERRIDE=${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
export STACK_ALLOW_HAND_CONTACT=${STACK_ALLOW_HAND_CONTACT:-1}
export STACK_FIXED_BOX_SIZE_IDS=${STACK_VIEW_BOX_IDS:-${STACK_FIXED_BOX_SIZE_IDS:-}}
export STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.15}
export STACK_TOP_FOLLOWS_BOTTOM=${STACK_TOP_FOLLOWS_BOTTOM:-1}

bash "$LOCAL_VIEW" "$@"
