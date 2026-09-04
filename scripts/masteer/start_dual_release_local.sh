#!/bin/bash
# Start dual-support training in a new session so it survives this terminal/Codex session.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms19_dualsupport_release_c06_s0}
LOG="$ROOT/runs/queue/logs/$TAG.log"
PID_FILE="$ROOT/runs/queue/logs/$TAG.pid"
OUT_DIR="$ROOT/TokenHSI-masteer/output/masteer/$TAG"

if [ -e "$OUT_DIR" ] || [ -e "$LOG" ] || [ -e "$PID_FILE" ]; then
    echo "같은 tag의 출력/log/pid가 이미 있다: $TAG" >&2
    echo "기존 결과를 덮어쓰지 않으므로 새 tag를 사용한다." >&2
    exit 3
fi

mkdir -p "$(dirname "$LOG")"
nohup setsid bash "$ROOT/scripts/masteer/train_dual_release_local.sh" "$TAG" \
    > "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

echo "started tag=$TAG pid=$pid"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
