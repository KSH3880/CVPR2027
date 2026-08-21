#!/bin/bash
# 손으로 돌리는 것(스모크·회귀)이 GPU 자리를 잡는 도구.
# autofill 이 자리를 셀 때 이 락을 같이 세므로, 이걸 안 쓰면 데몬이 그 위에 얹는다.
#
#   bash scripts/ma/gpu.sh hold 6        한 자리 잡는다
#   bash scripts/ma/gpu.sh hold 6 2      두 자리 = 그 카드를 통째로 막는다
#   bash scripts/ma/gpu.sh free 6
#   bash scripts/ma/gpu.sh status
#
# 락 파일 이름은 gpu<N>.<무엇이든> 이어야 한다. autofill 이 "gpu*.*" 로 세기 때문에
# 점이 없는 이름(gpu6)은 세어지지 않는다.
set -u
LOCKS=/home/hwanhee/CVPR2027/runs/queue/gpu_locks
mkdir -p "$LOCKS"

case "${1:-status}" in
hold)
    g=${2:?gpu index}; n=${3:-1}
    for i in $(seq 1 "$n"); do
        setsid sleep infinity >/dev/null 2>&1 &
        echo $! > "$LOCKS/gpu$g.hold$!"
        echo "gpu$g 자리 잡음 (holder pid $!)"
    done
    ;;
free)
    g=${2:?gpu index}
    shopt -s nullglob
    for f in "$LOCKS/gpu$g".hold*; do
        kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; echo "$(basename "$f") 해제"
    done
    ;;
status)
    shopt -s nullglob
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    while IFS=, read -r idx used; do
        idx=${idx// /}; used=${used// /}
        n=0; held=0
        for f in "$LOCKS/gpu$idx".*; do
            pid=$(cat "$f" 2>/dev/null)
            [ -n "$pid" ] && [ -d "/proc/$pid" ] || { rm -f "$f"; continue; }
            n=$((n+1)); case "$f" in *.hold*) held=$((held+1));; esac
        done
        printf "gpu%-2s %7s MiB  락 %s (손으로 잡은 것 %s)\n" "$idx" "$used" "$n" "$held"
    done
    ;;
*)  echo "usage: gpu.sh {hold <n> [count]|free <n>|status}"; exit 1;;
esac
