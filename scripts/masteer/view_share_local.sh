#!/bin/bash
# GPU-backed headless Xorg의 한 viewer를 두 noVNC 포트로 공유한다.
# 기본: 본인 6109, 인턴 6100, compute GPU 1, graphics GPU 0, DISPLAY :42.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?사용법: view_share_local.sh <tag 또는 ckpt경로> [env수]}
ENVS=${ENVS:-${2:-1}}
DISPLAY=${HEADLESS_DISPLAY:-:42}
PRIMARY_PORT=${PORT:-6109}
GUEST_PORT=${GUEST_PORT:-6100}
VNC_PORT=${VNC_PORT:-5950}
NOVNC_DIR=${TOKENHSI_NOVNC_DIR:-/usr/share/novnc}
X11VNC=${TOKENHSI_X11VNC:-$(command -v x11vnc || true)}
WEBSOCKIFY=${TOKENHSI_WEBSOCKIFY:-$(command -v websockify || true)}

[ -n "$X11VNC" ] || { echo "x11vnc를 찾을 수 없다" >&2; exit 2; }
[ -n "$WEBSOCKIFY" ] || { echo "websockify를 찾을 수 없다" >&2; exit 2; }
[ -d "$NOVNC_DIR" ] || { echo "noVNC 디렉터리 없음: $NOVNC_DIR" >&2; exit 2; }
if ! xdpyinfo -display "$DISPLAY" 2>/dev/null | grep -w DRI3 >/dev/null; then
    echo "$DISPLAY가 실행 중이지 않거나 DRI3를 제공하지 않는다" >&2
    echo "확인: systemctl status tokenhsi-headless-xorg.service" >&2
    exit 2
fi

for port in "$PRIMARY_PORT" "$GUEST_PORT" "$VNC_PORT"; do
    if ss -tln 2>/dev/null | grep -q ":$port "; then
        echo "포트 $port가 이미 사용 중이다" >&2
        exit 2
    fi
done

cleanup() {
    kill ${WEB_PRIMARY_PID:-} ${WEB_GUEST_PID:-} ${VNC_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$X11VNC" -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -nopw \
    -forever -shared -noxdamage -noshm -quiet &
VNC_PID=$!
sleep 1
"$WEBSOCKIFY" --web="$NOVNC_DIR" "127.0.0.1:$PRIMARY_PORT" "127.0.0.1:$VNC_PORT" \
    >/dev/null 2>&1 &
WEB_PRIMARY_PID=$!
"$WEBSOCKIFY" --web="$NOVNC_DIR" "127.0.0.1:$GUEST_PORT" "127.0.0.1:$VNC_PORT" \
    >/dev/null 2>&1 &
WEB_GUEST_PID=$!
sleep 1

echo "=============================================================="
echo " 본인       http://localhost:$PRIMARY_PORT/vnc.html"
echo " 인턴       ssh -L $GUEST_PORT:localhost:$GUEST_PORT inkyu@100.89.9.110"
echo "            http://localhost:$GUEST_PORT/vnc.html"
echo " 공유화면   DISPLAY=$DISPLAY  VNC=$VNC_PORT  (-shared)"
echo "=============================================================="

DISPLAY=$DISPLAY MA_GPU=${MA_GPU:-1} VIEW_GRAPHICS_GPU=${VIEW_GRAPHICS_GPU:-0} \
    ENVS=$ENVS bash "$ROOT/scripts/masteer/view_local.sh" "$TAG"
