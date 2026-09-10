#!/bin/bash
# 로컬 Isaac Gym viewer: frozen ms18 executor + 별도 C1 coordinator PTH.
#
# COORD_CKPT=runs/coord/checkpoints/c1.pth MS_VIZ=7 ENVS=1 MS_CAM=top \
#   bash scripts/coord/view_local.sh ms18_maskteam_origscale_c06_s0
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-coord"
EXEC_REPO=${COORD_EXEC_REPO:-$ROOT/TokenHSI-masteer}
INPUT=${1:?사용법: view_local.sh <ms18 tag 또는 PTH> [env수]}
ENVS=${ENVS:-${2:-1}}

if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ENVS는 양의 정수여야 한다: $ENVS" >&2
    exit 2
fi

resolve_file() {
    local value=$1 candidate
    for candidate in "$value" "$ROOT/$value" "$REPO/$value" "$EXEC_REPO/$value"; do
        if [ -f "$candidate" ]; then
            realpath "$candidate"
            return 0
        fi
    done
    return 1
}

if EXEC_CKPT=$(resolve_file "$INPUT"); then
    NAME=$(basename "$(dirname "$(dirname "$EXEC_CKPT")")")
    SOURCE_TAG=""
    case "$EXEC_CKPT" in
        "$EXEC_REPO/output/masteer/"*)
            SOURCE_TAG=${EXEC_CKPT#"$EXEC_REPO/output/masteer/"}
            SOURCE_TAG=${SOURCE_TAG%%/*}
            ;;
    esac
else
    NAME=$INPUT
    SOURCE_TAG=$INPUT
    EXEC_CKPT=$(find "$EXEC_REPO/output/masteer/$NAME" -type f \
        \( -name 'Humanoid.pth' -o -name 'Humanoid_*.pth' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -z "$EXEC_CKPT" ]; then
        echo "ms18 체크포인트 없음: $EXEC_REPO/output/masteer/$NAME" >&2
        exit 1
    fi
fi

if [ -z "${COORD_CKPT:-}" ]; then
    echo "COORD_CKPT=<coordinator.pth>를 지정한다." >&2
    exit 1
fi
COORD_CKPT_INPUT=$COORD_CKPT
if ! COORD_CKPT=$(resolve_file "$COORD_CKPT_INPUT"); then
    echo "coordinator 체크포인트 없음: $COORD_CKPT_INPUT" >&2
    exit 1
fi
export COORD_CKPT

# Select the checkpoint loader from the coordinator run sidecar. Previously
# the viewer silently defaulted to C1, so a valid C2 checkpoint failed with a
# misleading schema/candidate_k mismatch unless COORD_MODEL=c2 was typed by
# hand. Explicit user overrides still win.
if [ -z "${COORD_MODEL:-}" ]; then
    case "$COORD_CKPT" in
        "$ROOT/runs/coord/"*)
            COORD_REL=${COORD_CKPT#"$ROOT/runs/coord/"}
            COORD_SOURCE_TAG=${COORD_REL%%/*}
            COORD_ENV_FILE="$ROOT/runs/queue/logs/$COORD_SOURCE_TAG.env"
            if [ -f "$COORD_ENV_FILE" ]; then
                COORD_MODEL=$(bash -c \
                    'source "$1"; printf "%s" "${COORD_MODEL:-}"' \
                    _ "$COORD_ENV_FILE")
            fi
            ;;
    esac
fi
export COORD_MODEL=${COORD_MODEL:-c1}

if [ -z "${MA_TOKEN+x}" ] && [ -n "$SOURCE_TAG" ]; then
    ENV_FILE="$ROOT/runs/queue/logs/$SOURCE_TAG.env"
    if [ -f "$ENV_FILE" ]; then
        MA_TOKEN=$(bash -c 'source "$1"; printf "%s" "${MA_TOKEN:-mask}"' _ "$ENV_FILE")
        export MA_TOKEN
    fi
fi

if [ -n "${MS_CKPT:-}" ]; then
    if ! BASE_CKPT=$(resolve_file "$MS_CKPT"); then
        echo "stage1 체크포인트 없음: $MS_CKPT" >&2
        exit 1
    fi
else
    BASE_CKPT=""
    for candidate in \
        "$REPO/output/tokenhsi/ckpt_stage1.pth" \
        "$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth" \
        "$ROOT/../TokenHSI/output/tokenhsi/ckpt_stage1.pth"; do
        if [ -f "$candidate" ]; then
            BASE_CKPT=$(realpath "$candidate")
            break
        fi
    done
    if [ -z "$BASE_CKPT" ]; then
        echo "stage1 체크포인트 없음. MS_CKPT=<ckpt_stage1.pth>를 지정한다." >&2
        exit 1
    fi
fi

if [ "${LOCAL_HEADLESS:-0}" = 0 ] && [ -z "${DISPLAY:-}" ]; then
    echo "DISPLAY가 없다. 데스크톱 터미널에서 실행하거나 LOCAL_HEADLESS=1을 쓴다." >&2
    exit 1
fi

if [ -n "${TOKENHSI_DATA_ROOT:-}" ]; then
    DATA_ROOT=$(realpath "$TOKENHSI_DATA_ROOT")
else
    DATA_ROOT=""
    for candidate in "$ROOT/../TokenHSI/tokenhsi/data" "$ROOT/TokenHSI/tokenhsi/data"; do
        if [ -d "$candidate/dataset_sit/objects" ]; then
            DATA_ROOT=$(realpath "$candidate")
            break
        fi
    done
fi
for relative in \
    dataset_amass_loco/motions dataset_sit/motions dataset_sit/objects \
    dataset_carry/motions dataset_amass_climb/motions dataset_amass_climb/objects; do
    target="$REPO/tokenhsi/data/$relative"
    source_dir="$DATA_ROOT/$relative"
    if [ -L "$target" ] && [ ! -e "$target" ]; then
        echo "깨진 데이터 링크: $target" >&2
        exit 1
    fi
    if [ -e "$target" ]; then
        continue
    fi
    if [ -z "$DATA_ROOT" ] || [ ! -d "$source_dir" ]; then
        echo "coord 데이터 없음: $relative" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$target")"
    ln -s "$source_dir" "$target"
done

. "$ROOT/scripts/masteer/viz_env.sh"
viz_expand || exit 1

export MS_TASK=HumanoidMACoordCarry
export COORD_VIEWER=1
export COORD_PROVIDER=${COORD_PROVIDER:-learned}
export COORD_REPLAN_STEPS=${COORD_REPLAN_STEPS:-6}
export COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75}
export COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_SPEED_LIMIT=${COORD_SPEED_LIMIT:-1}
export COORD_DRAW_CANDIDATES=${COORD_DRAW_CANDIDATES:-1}
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_TOKEN=${MA_TOKEN:-mask}
export MA_SEP=${MA_SEP:-0}
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=0
export MS_M_LO=${MS_M_LO:-0.25}
export MS_ZERO=${MS_ZERO:-0}
export MS_SCEN=${MS_SCEN:-free}
export MS_CLIP=${MS_CLIP:-1}
export MS_DBG=${MS_DBG:-1}
export MS_LAT_MAX=${MS_LAT_MAX:-2.2}
export MS_DRAW_SPEED=${MS_DRAW_SPEED:-1}
export MS_CAM=${MS_CAM:-top}
export MS_CAM_H=${MS_CAM_H:-17}
export MS_CAM_B=${MS_CAM_B:-9}
export MS_SEED=${MS_SEED:-0}
export MS_VIEW_TIMED_CROSS=${MS_VIEW_TIMED_CROSS:-0}
export MS_VIEW_TIMED_CROSS_PROB=${MS_VIEW_TIMED_CROSS_PROB:-1.0}
export MS_VIEW_TIMED_CROSS_TOL=${MS_VIEW_TIMED_CROSS_TOL:-0.25}

if [ "$MS_VIEW_TIMED_CROSS" != 0 ] && [ "$MS_SCEN" != cross ]; then
    echo "MS_VIEW_TIMED_CROSS=1은 MS_SCEN=cross와 함께 사용한다." >&2
    exit 2
fi

if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    elif [ -f /home/cvlab/anaconda3/etc/profile.d/conda.sh ]; then
        CONDA_BASE=/home/cvlab/anaconda3
    else
        echo "conda를 찾지 못했다. CONDA_BASE=<conda root>를 지정한다." >&2
        exit 1
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"

SAFE_NAME=$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9_.-' '_')
CFG_SRC="$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
CFG=$(mktemp "/tmp/coord_view_${SAFE_NAME}.XXXXXX.yaml")
SNAP_DIR=$(mktemp -d "/tmp/coord_view_${SAFE_NAME}.XXXXXX")
mkdir -p "$SNAP_DIR/nn"
SNAP="$SNAP_DIR/nn/Humanoid.pth"
cleanup() {
    rm -f -- "$CFG"
    rm -rf -- "$SNAP_DIR"
}
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" "$CFG_SRC" > "$CFG"
cp -- "$EXEC_CKPT" "$SNAP"

RUN_ARGS=(--test --eval_task carry)
if [ "${MS_EVAL:-0}" != 0 ]; then RUN_ARGS+=(--eval); fi
if [ "${LOCAL_HEADLESS:-0}" != 0 ]; then RUN_ARGS+=(--headless); fi

echo "=============================================================="
echo " repo         $REPO"
echo " frozen ms18  $EXEC_CKPT"
echo " coordinator  $COORD_CKPT"
echo " provider     $COORD_PROVIDER  model=$COORD_MODEL  replan=$COORD_REPLAN_STEPS"
echo " env          $ENVS x 2명  CLIP=$MS_CLIP  camera=$MS_CAM"
echo " timed cross  $MS_VIEW_TIMED_CROSS  prob=$MS_VIEW_TIMED_CROSS_PROB tol=${MS_VIEW_TIMED_CROSS_TOL}s"
echo "=============================================================="

cd "$REPO"
python -u ./tokenhsi/run.py \
    --task "$MS_TASK" \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$BASE_CKPT" \
    --checkpoint "$SNAP" \
    --num_envs "$ENVS" --seed "$MS_SEED" "${RUN_ARGS[@]}"
