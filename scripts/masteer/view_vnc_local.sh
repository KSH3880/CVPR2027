#!/bin/bash
# GPU 1 기반 headless Xorg(:43)에서 viewer 하나를 만들고 그 창만 noVNC로 공개한다.
# 동시에 여러 번 실행해도 각 포트가 이번 실행에서 새로 생긴 창만 잡는다.
#
#   PORT=6109 MA_GPU=1 bash scripts/masteer/view_vnc_local.sh <tag> [env수]
#   PORT=6100 MA_GPU=1 bash scripts/masteer/view_vnc_local.sh <tag> [env수]
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?사용법: view_vnc_local.sh <tag 또는 ckpt경로> [env수]}
ENVS=${ENVS:-${2:-1}}
VIEW_DISPLAY=${VIEW_DISPLAY:-:43}
PORT=${PORT:-6109}
VNC_PORT=${VNC_PORT:-$((5900 + PORT - 6059))}
WINDOW_TITLE=${VNC_WINDOW_TITLE:-Isaac Gym}
NOVNC_DIR=${TOKENHSI_NOVNC_DIR:-/usr/share/novnc}
X11VNC=${TOKENHSI_X11VNC:-$(command -v x11vnc || true)}
WEBSOCKIFY=${TOKENHSI_WEBSOCKIFY:-$(command -v websockify || true)}

for command_name in xdpyinfo xwininfo xprop flock ss; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "필수 명령을 찾을 수 없다: $command_name" >&2
        exit 2
    }
done
[ -n "$X11VNC" ] || { echo "x11vnc를 찾을 수 없다" >&2; exit 2; }
[ -n "$WEBSOCKIFY" ] || { echo "websockify를 찾을 수 없다" >&2; exit 2; }
[ -d "$NOVNC_DIR" ] || { echo "noVNC 디렉터리 없음: $NOVNC_DIR" >&2; exit 2; }

if ! xdpyinfo -display "$VIEW_DISPLAY" >/dev/null 2>&1; then
    echo "GPU-backed Xorg가 준비되지 않았다: DISPLAY=$VIEW_DISPLAY" >&2
    echo "확인: systemctl status tokenhsi-headless-xorg.service" >&2
    exit 2
fi
if ! xdpyinfo -display "$VIEW_DISPLAY" 2>/dev/null | grep -w DRI3 >/dev/null; then
    echo "DISPLAY=$VIEW_DISPLAY에 DRI3가 없다" >&2
    exit 2
fi

for port in "$PORT" "$VNC_PORT"; do
    if ss -tln 2>/dev/null | grep -q ":$port "; then
        echo "포트 $port가 이미 사용 중이다" >&2
        exit 2
    fi
done

STATE_DIR=$(mktemp -d "$ROOT/runs/view_${PORT}.XXXXXX")
OLD_WINDOWS="$STATE_DIR/old_windows"
VIEW_LOG="$STATE_DIR/view.log"
VNC_LOG="$STATE_DIR/x11vnc.log"
WEB_LOG="$STATE_DIR/websockify.log"

cleanup() {
    kill ${WEB_PID:-} ${VNC_PID:-} ${VIEW_PID:-} 2>/dev/null || true
    if [ -n "${VIEW_PID:-}" ]; then
        kill -- -"$VIEW_PID" 2>/dev/null || true
    fi
    wait ${WEB_PID:-} ${VNC_PID:-} ${VIEW_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

list_windows() {
    DISPLAY="$VIEW_DISPLAY" XAUTHORITY="${XAUTHORITY:-}" \
        xwininfo -root -tree 2>/dev/null | \
        awk -v title="$WINDOW_TITLE" 'index($0, "\"" title) {print $1}'
}

# 두 launcher가 동시에 기존 창 목록을 읽고 같은 새 창을 선택하지 않도록
# 창 생성과 선택이 끝날 때까지 짧게 직렬화한다.
LOCK_FILE=${XDG_RUNTIME_DIR:-/tmp}/tokenhsi-masteer-view-window.lock
exec 9>"$LOCK_FILE"
flock -x 9
list_windows >"$OLD_WINDOWS"

setsid env DISPLAY="$VIEW_DISPLAY" ENVS="$ENVS" MA_GPU=${MA_GPU:-1} \
    VIEW_GRAPHICS_GPU=${VIEW_GRAPHICS_GPU:-1} \
    TOKENHSI_USE_SYSTEM_VULKAN=1 \
    bash "$ROOT/scripts/masteer/view_local.sh" "$TAG" >"$VIEW_LOG" 2>&1 &
VIEW_PID=$!

WINDOW_ID=""
for _ in $(seq 1 480); do
    if ! kill -0 "$VIEW_PID" 2>/dev/null; then
        wait "$VIEW_PID" || true
        echo "viewer가 창을 만들기 전에 종료됐다. 로그: $VIEW_LOG" >&2
        tail -80 "$VIEW_LOG" >&2 || true
        exit 1
    fi
    while IFS= read -r candidate; do
        if ! grep -Fxq "$candidate" "$OLD_WINDOWS"; then
            WINDOW_ID=$candidate
            break
        fi
    done < <(list_windows)
    [ -n "$WINDOW_ID" ] && break
    sleep 0.25
done
[ -n "$WINDOW_ID" ] || {
    echo "120초 안에 새 '$WINDOW_TITLE' 창을 찾지 못했다. 로그: $VIEW_LOG" >&2
    exit 1
}

# 사람이 xwininfo로 보더라도 어느 포트의 창인지 구분되게 이름도 바꾼다.
UNIQUE_TITLE="$WINDOW_TITLE [$PORT]"
DISPLAY="$VIEW_DISPLAY" XAUTHORITY="${XAUTHORITY:-}" \
    xprop -id "$WINDOW_ID" -f _NET_WM_NAME 8u -set _NET_WM_NAME "$UNIQUE_TITLE" \
    >/dev/null 2>&1 || true
DISPLAY="$VIEW_DISPLAY" XAUTHORITY="${XAUTHORITY:-}" \
    xprop -id "$WINDOW_ID" -f WM_NAME 8s -set WM_NAME "$UNIQUE_TITLE" \
    >/dev/null 2>&1 || true

env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
    "$X11VNC" -display "$VIEW_DISPLAY" -id "$WINDOW_ID" -waitmapped \
    -rfbport "$VNC_PORT" -localhost -nopw -forever -shared \
    -noxdamage -noshm -quiet >"$VNC_LOG" 2>&1 &
VNC_PID=$!

for _ in $(seq 1 80); do
    ss -tln 2>/dev/null | grep -q ":$VNC_PORT " && break
    kill -0 "$VNC_PID" 2>/dev/null || {
        echo "x11vnc가 종료됐다. 로그: $VNC_LOG" >&2
        exit 1
    }
    sleep 0.25
done
ss -tln 2>/dev/null | grep -q ":$VNC_PORT " || {
    echo "VNC 포트가 열리지 않았다: $VNC_PORT" >&2
    exit 1
}

"$WEBSOCKIFY" --web="$NOVNC_DIR" "127.0.0.1:$PORT" \
    "127.0.0.1:$VNC_PORT" >"$WEB_LOG" 2>&1 &
WEB_PID=$!
sleep 1
kill -0 "$WEB_PID" 2>/dev/null || {
    echo "noVNC가 종료됐다. 로그: $WEB_LOG" >&2
    exit 1
}

flock -u 9
echo "=============================================================="
echo " noVNC      http://localhost:$PORT/vnc.html"
echo " X 창       DISPLAY=$VIEW_DISPLAY  id=$WINDOW_ID  title=$UNIQUE_TITLE"
echo " GPU        compute=${MA_GPU:-1}  graphics=${VIEW_GRAPHICS_GPU:-1}"
echo " 다른 PC   ssh -L $PORT:localhost:$PORT inkyu@100.89.9.110"
echo " 로그       $VIEW_LOG"
echo " 종료       Ctrl-C 또는 Isaac Gym 창에서 ESC"
echo "=============================================================="

set +e
wait "$VIEW_PID"
VIEW_RC=$?
set -e
exit "$VIEW_RC"
