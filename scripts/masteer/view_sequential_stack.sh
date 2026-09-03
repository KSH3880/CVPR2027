#!/bin/bash
# noVNC viewer for the strict A1-place -> A1-retreat -> A2-stack task.
# Existing scripts/masteer/view.sh owns Xvfb, VNC, cfg generation and cleanup.
#
# Usage:
#   MA_GPU=7 PORT=6100 bash scripts/masteer/view_sequential_stack.sh
#   MA_GPU=7 PORT=6101 bash scripts/masteer/view_sequential_stack.sh /abs/policy.pth 1
#
# MS_CKPT, when supplied, must be the TokenHSI stage1 backbone checkpoint,
# not the single-task carry checkpoint.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
POLICY=${1:-"$REPO/output/ms18_maskteam_origscale_c06_s0_00009000.pth"}
ENVS=${ENVS:-${2:-1}}

if [ ! -f "$POLICY" ]; then
    echo "정책 체크포인트 없음: $POLICY" >&2
    exit 1
fi
if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ENVS는 양의 정수여야 한다: $ENVS" >&2
    exit 2
fi
if [ -n "${MS_CKPT:-}" ] && [ ! -f "$MS_CKPT" ]; then
    echo "stage1 체크포인트 없음: $MS_CKPT" >&2
    exit 1
fi
case "${MS_CKPT:-}" in
    *ckpt_carry.pth)
        echo "MS_CKPT에 single-task carry checkpoint를 지정하면 안 된다." >&2
        echo "MS_CKPT를 빼거나 ckpt_stage1.pth 경로를 지정한다." >&2
        exit 2
        ;;
esac

export MS_TASK=HumanoidMASequentialStackCarry
export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=${MS_CLIP:-1}
export MS_SCEN=${MS_SCEN:-free}
export MS_DBG=${MS_DBG:-0}
export STACK_TASK_MODE=stack
export STACK_VIRTUAL_RETREAT_BOX=${STACK_VIRTUAL_RETREAT_BOX:-1}
export STACK_DEBUG=${STACK_DEBUG:-1}
export STACK_RETREAT_RELEASE_TOL=${STACK_RETREAT_RELEASE_TOL:-0.30}
export STACK_RETREAT_MIN_PROGRESS=${STACK_RETREAT_MIN_PROGRESS:-0.60}
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-1200}
export STACK_GROUND_Z=${STACK_GROUND_Z:-0.0}
export ENVS

echo "sequential stack: A1 place -> A1 retreat -> A2 carry-to-top"
echo "policy: $POLICY"
echo "viewer: http://localhost:${PORT:-6100}/vnc.html"

exec bash "$ROOT/scripts/masteer/view.sh" "$POLICY" "$ENVS"
