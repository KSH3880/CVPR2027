#!/bin/bash
# ms18 계열 masteer 체크포인트를 단일 humanoid의 steering-only(traj)로 본다.
#
# 기존 view.sh를 수정하거나 별도 구현으로 복제하지 않는다. 실행할 때 /tmp 복사본의
# eval task 한 줄만 carry -> traj로 바꿔서, 원본의 Vulkan GPU guard/VNC/스냅샷
# 동작을 그대로 사용한다.
#
# 사용법:
#   MA_GPU=7 PORT=6100 MS_VIZ=1_follow_curve \
#     bash scripts/masteer/view_only_steering.sh <tag-or-pth> [envs]
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
VIEW="$ROOT/scripts/masteer/view.sh"
INPUT=${1:?사용법: view_only_steering.sh <tag 또는 ckpt경로> [env수]}

# view.sh의 계약이 바뀌면 조용히 carry로 실행하지 말고 즉시 거부한다.
if [[ $(grep -Fxc 'VIEW_ARGS=(--test --eval_task carry)' "$VIEW") != 1 ]]; then
    echo "view_only_steering: view.sh의 eval_task 지점을 정확히 찾지 못했다" >&2
    exit 2
fi

TMP=$(mktemp "/tmp/masteer_view_only_steering.XXXXXX.sh")
cleanup() {
    rm -f -- "$TMP"
}
trap cleanup EXIT INT TERM

# /tmp에서 실행해도 ROOT가 원래 저장소를 가리키도록 고정한다.
sed \
    -e "s|^ROOT=.*|ROOT=$ROOT|" \
    -e 's/VIEW_ARGS=(--test --eval_task carry)/VIEW_ARGS=(--test --eval_task traj)/' \
    "$VIEW" > "$TMP"

# A=1이어야 MA의 carry 전용 강제가 풀려 traj가 선택된다. ms75는 MA_TOKEN=mask다.
export MS_SINGLE=1
export MA_TOKEN=mask
export MS_TASK=HumanoidMAOnlySteering
export MS_EVAL=1
export MS_STACK_START=0
export MS_CLIP=1
export MS_ENDCLAMP=0

bash "$TMP" "$@"
