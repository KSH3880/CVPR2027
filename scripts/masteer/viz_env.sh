# MS_VIZ 하나를 구체 변수로 편다. view.sh 와 record.sh 가 **둘 다 이 파일을 source
# 한다** -- 매핑이 두 곳에 있으면 뷰어에서 본 것과 영상이 어긋난다.
#
#   MS_VIZ=<번호>_<이름>_<straight|curve>      값이 곧 영상 파일명이다
#
# 호출자가 직접 준 값이 항상 이긴다 (`${VAR:=..}` 는 비어 있을 때만 채운다).
# 그래서 MS_VIZ=4_cross_yield_curve MS_DT=1.0 처럼 일부만 덮어쓸 수 있다.

VIZ_ALL="1_follow_straight 1_follow_curve
2_speed_straight 2_speed_curve
3_cross_hit_straight 3_cross_hit_curve
4_cross_yield_straight 4_cross_yield_curve
5_stop_straight 5_stop_curve
6_parallel_straight 6_parallel_curve"

# 번호 1~12 로도 부를 수 있다. **홀수 = straight, 짝수 = curve.**
# 입력만 번호고, MS_VIZ 는 곧바로 이름으로 바뀐다 -- 파일명은 항상 이름이다
# (`7.mp4` 같은 폴더는 발표에 못 쓴다).
viz_num() {
    case "$1" in
        1)  echo 1_follow_straight ;;      2)  echo 1_follow_curve ;;
        3)  echo 2_speed_straight ;;       4)  echo 2_speed_curve ;;
        5)  echo 3_cross_hit_straight ;;   6)  echo 3_cross_hit_curve ;;
        7)  echo 4_cross_yield_straight ;; 8)  echo 4_cross_yield_curve ;;
        9)  echo 5_stop_straight ;;        10) echo 5_stop_curve ;;
        11) echo 6_parallel_straight ;;    12) echo 6_parallel_curve ;;
        *)  return 1 ;;
    esac
}

viz_expand() {
    # `[ .. ] && return` 형태로 쓰지 않는다 -- 호출부가 `set -e` 라 && 예외 규칙에
    # 기대게 되고, 그건 bash 판이나 옵션이 바뀌면 조용히 깨진다.
    if [ -z "${MS_VIZ:-}" ]; then
        return 0
    fi

    case "$MS_VIZ" in
        [0-9]|[0-9][0-9])
            _n=$(viz_num "$MS_VIZ") || {
                echo "MS_VIZ 번호는 1~12 다: $MS_VIZ" >&2; return 1; }
            MS_VIZ=$_n ;;
    esac

    case "$MS_VIZ" in
        1_follow_*)      : "${MS_MRAND:=0}"; : "${MA_SEP:=9}" ;;            # 속도 고정 1.5 m/s
        2_speed_*)       : "${MS_MRAND:=4}"; : "${MA_SEP:=9}" ;;            # 4구간 속도 명령
        3_cross_hit_*)   : "${MS_SCEN:=cross}"; : "${MS_DT:=0}" ;;          # 감속 없음 -> 동시 도착
        4_cross_yield_*) : "${MS_SCEN:=cross}"; : "${MS_DT:=2.0}" ;;        # a1 국소 감속
        5_stop_*)        : "${MS_M_LO:=0}"; : "${MS_MRAND:=4}"; : "${MA_SEP:=9}" ;;
        6_parallel_*)    : "${MS_SCEN:=parallel}"; : "${MS_GAP:=1.0}" ;;
        *) echo "알 수 없는 MS_VIZ: $MS_VIZ" >&2
           echo "쓸 수 있는 값:" >&2; echo "$VIZ_ALL" | sed 's/^/  /' >&2
           return 1 ;;
    esac

    case "$MS_VIZ" in
        *_straight) : "${MS_LAT_MAX:=0}" ;;   # 직선
        # **MS_SCEN_CURVE=1 이 반드시 같이 가야 한다.** _scen_env 는 시나리오일 때
        # MS_LAT_MAX 를 0 으로 덮어써서, 이게 없으면 3/4/6 의 _curve 가 조용히
        # 직선이 된다 (화면은 멀쩡하고 파일명만 curve 인 상태).
        *_curve)    : "${MS_SCEN_CURVE:=1}" ;;
        *) echo "MS_VIZ 는 _straight 또는 _curve 로 끝나야 한다: $MS_VIZ" >&2; return 1 ;;
    esac

    # **free 시나리오는 두 사람을 떼어놓는다.** 1~4·9~10 은 "서로 상관 안 주는
    # 상태"를 보이는 것이 목적인데 MA_SEP=0 이면 원점 반경 1.5 m 에 같이 뿌려져
    # 서로 밀친다. cross/parallel 은 _scen_env 가 MA_SEP 을 0 으로 강제하므로
    # 여기서 안 건드린다 (떼어놓으면 만나지를 못한다).
    export MS_SCEN MS_MRAND MS_DT MS_M_LO MS_GAP MS_LAT_MAX MS_SCEN_CURVE MA_SEP
    return 0
}
