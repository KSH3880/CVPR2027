#!/bin/bash
# Floor-start visual check. Usage: pickup_diag_view.sh <f22x2|tag> [gpu] [port]
# Set PICKUP_ITER=3000 to select Humanoid_00003000.pth instead of Humanoid.pth.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=${1:?usage: pickup_diag_view.sh <f22x2|tag> [gpu] [port]}
GPU=${2:-6}
PORT_NUM=${3:-6100}
case "$GPU" in 6|7) ;; *) echo "this diagnostic is restricted to GPU 6 or 7" >&2; exit 2 ;; esac

if [ "$NAME" = f22x2 ]; then
    TASK=HumanoidMAF22Carry
    CK=$ROOT/runs/checkpoints/steer/f22b_long_c20/Humanoid_00025000.pth
    export MA_TOKENIZER_ZERO=1 MA_SEP=9 MA_SPAWN_GAP=1.0
    export MA_C=0 MA_BETA=0
    export MS_SCEN=free MS_MRAND=0 MS_ZERO=0 MS_POS_C=2 MS_VEL_W=1 MS_VEL_K=5
    export MS_CLIP=0 MS_ENDCLAMP=0 MS_SEED=0
    export MS_PICK_HAND=0.35 MS_PICK_LIFT=0.10 MS_PICK_FRAMES=10
else
    ENV_FILE=$ROOT/runs/logs/masteer_pickup/$NAME.env
    OUT=$ROOT/runs/output/masteer_pickup/$NAME
    if [ ! -f "$ENV_FILE" ]; then
        echo "missing sidecar: $ENV_FILE" >&2
        exit 4
    fi
    if [ ! -d "$OUT" ]; then
        echo "missing training output: $OUT" >&2
        exit 4
    fi
    source "$ENV_FILE"
    TASK=HumanoidMASteerCarry
    if [ -n "${PICKUP_ITER:-}" ]; then
        case "$PICKUP_ITER" in
            *[!0-9]*) echo "PICKUP_ITER must be an integer: $PICKUP_ITER" >&2; exit 2 ;;
        esac
        printf -v CK_NAME 'Humanoid_%08d.pth' "$PICKUP_ITER"
        CK=$(find "$OUT" -name "$CK_NAME" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    else
        CK=$(find "$OUT" -name Humanoid.pth -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
fi
if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
    echo "missing checkpoint: ${CK:-unset}" >&2
    exit 4
fi

MS_TASK=$TASK MS_EVAL=0 MS_FLOOR_START=1 \
MS_SNAP_DIR=$ROOT/runs/tmp/masteer_pickup_view \
MA_GPU=$GPU PORT=$PORT_NUM ENVS=1 \
MS_VIZ=${MS_VIZ:-1_follow_straight} \
bash "$ROOT/scripts/masteer/view.sh" "$CK"
