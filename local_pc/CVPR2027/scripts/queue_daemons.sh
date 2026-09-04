#!/bin/bash
# 큐 데몬만 관리한다. 이미 분리 실행된 학습·평가 프로세스는 건드리지 않는다.
#
#   bash scripts/queue_daemons.sh start
#   bash scripts/queue_daemons.sh status
#   bash scripts/queue_daemons.sh stop
#   bash scripts/queue_daemons.sh restart
set -u
ROOT=/home/hwanhee/CVPR2027
TRACKS="steer ma lab masteer"
cd "$ROOT"

pid_of() {
    pgrep -f -- "python3 scripts/autofill.py --track $1 --loop" | head -1
}

validate() {
    python3 scripts/check_docs.py || {
        echo "문서·큐 검증 실패: 데몬 상태를 변경하지 않음" >&2
        return 1
    }
}

start_one() {
    local track=$1 pid log
    pid=$(pid_of "$track")
    if [ -n "$pid" ]; then
        echo "$track 이미 돌고 있음 (pid $pid)"
        return
    fi
    log="runs/queue/autofill_${track}.log"
    setsid nohup python3 scripts/autofill.py --track "$track" --loop >> "$log" 2>&1 < /dev/null &
    disown
    sleep 2
    pid=$(pid_of "$track")
    [ -n "$pid" ] && echo "$track 시작 (pid $pid)  로그 $log" || {
        echo "$track 시작 실패: $log 확인" >&2
        return 1
    }
}

stop_one() {
    local track=$1 pid attempt
    pid=$(pid_of "$track")
    if [ -z "$pid" ]; then
        echo "$track 안 돌고 있음"
        return
    fi
    kill "$pid"
    for attempt in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || {
            echo "$track 정지 (pid $pid)"
            return
        }
        sleep 0.2
    done
    echo "$track 정지 확인 실패 (pid $pid); 강제 종료하지 않음" >&2
    return 1
}

case "${1:-status}" in
start)
    validate || exit 1
    for track in $TRACKS; do start_one "$track" || exit 1; done
    ;;
stop)
    for track in $TRACKS; do stop_one "$track" || exit 1; done
    ;;
restart)
    validate || exit 1
    for track in $TRACKS; do stop_one "$track" || exit 1; done
    for track in $TRACKS; do start_one "$track" || exit 1; done
    ;;
status)
    for track in $TRACKS; do
        pid=$(pid_of "$track")
        printf "%-6s %s\n" "$track" "${pid:-정지됨}"
    done
    ;;
*)
    echo "usage: queue_daemons.sh {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
