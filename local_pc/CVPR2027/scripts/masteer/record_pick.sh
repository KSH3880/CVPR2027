#!/bin/bash
# 시드를 바꿔가며 찍어 **두 사람이 실제로 움직인 것**을 고른다.
#
# env 하나로 찍으면 초기 모션 샘플 하나에 성패가 걸린다 (교차 배치 성공률 0.53).
# 실제로 시드 0·1·3 이 연속으로 굳은 영상이 나왔다. 화면 변화로 판정해 다시 찍는다.
#
#   MS_VIZ=8 bash scripts/masteer/record_pick.sh ms4_m4_s0
set -u
TAG=${1:?사용법: record_pick.sh <tag>}
# record.sh·view.sh 와 **같은 매핑**을 쓴다. 여기만 빠지면 파일명이 갈라진다.
. /home/hwanhee/CVPR2027/scripts/masteer/viz_env.sh
viz_expand || exit 1
OUT=${OUT:-${MS_VIZ:?OUT 또는 MS_VIZ 필요}}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
THR=${THR:-0.6}
V=/home/hwanhee/CVPR2027/runs/results/video/0821
for sd in $SEEDS; do
    OUT=${OUT} MS_SEED=$sd MS_CAM=top bash /home/hwanhee/CVPR2027/scripts/masteer/record.sh "$TAG" \
        >/dev/null 2>&1
    m=$(python3 /home/hwanhee/CVPR2027/scripts/masteer/motion.py "$V/${OUT}.mp4" 2>/dev/null)
    echo "  seed $sd -> 변화 ${m:-?}%"
    awk -v a="${m:-0}" -v b="$THR" 'BEGIN{exit !(a>b)}' && { echo "  채택: seed $sd"; exit 0; }
done
echo "  모든 시드 실패 -- 마지막 것을 남긴다"
