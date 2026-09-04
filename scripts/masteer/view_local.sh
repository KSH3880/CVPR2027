#!/bin/bash
# 로컬 데스크톱의 Isaac Gym 창으로 masteer 정책을 본다.
# 서버용 view.sh와 달리 Xvfb/VNC/noVNC/PORT를 사용하지 않는다.
#
#   MS_VIZ=1 ENVS=1 MS_CAM=top bash scripts/masteer/view_local.sh ms12_base4L_s0
#   MS_VIZ=4 bash scripts/masteer/view_local.sh \
#     TokenHSI-masteer/output/masteer/ms12_base4L_s0/Humanoid_21-14-39-16/nn/Humanoid.pth
#
# 기본값:
#   ENVS=1, MS_CAM=top, MS_CLIP=1, conda env=tokenhsi
# 선택값:
#   TOKENHSI_CONDA_ENV=<env>, CONDA_BASE=<conda root>, MS_CKPT=<stage1 pth>
#   TOKENHSI_DATA_ROOT=<원본 TokenHSI의 tokenhsi/data>
#   LOCAL_HEADLESS=1  로컬 창 없이 기동 검증할 때만 사용
#   MS_EVAL=1         final evaluation과 같은 loco 시작만 볼 때 사용
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
INPUT=${1:?사용법: view_local.sh <tag 또는 ckpt경로> [env수]}
ENVS=${ENVS:-${2:-1}}

if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ENVS는 양의 정수여야 한다: $ENVS" >&2
    exit 2
fi

resolve_file() {
    local value=$1 candidate
    for candidate in "$value" "$ROOT/$value" "$REPO/$value"; do
        if [ -f "$candidate" ]; then
            realpath "$candidate"
            return 0
        fi
    done
    return 1
}

if CKPT=$(resolve_file "$INPUT"); then
    NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
    SOURCE_TAG=""
    case "$CKPT" in
        "$REPO/output/masteer/"*)
            SOURCE_TAG=${CKPT#"$REPO/output/masteer/"}
            SOURCE_TAG=${SOURCE_TAG%%/*}
            ;;
    esac
else
    NAME=$INPUT
    SOURCE_TAG=$INPUT
    CKPT=$(find "$REPO/output/masteer/$NAME" -type f -name Humanoid.pth \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -z "$CKPT" ]; then
        echo "체크포인트 없음: $REPO/output/masteer/$NAME" >&2
        exit 1
    fi
fi

# 태그로 실행할 때는 학습 당시 task와 reward/scenario 환경변수를 복원한다.
# 호출자가 명시한 환경변수(ENVS, MS_CAM 또는 실험 override)는 그대로 우선한다.
if [ -n "$SOURCE_TAG" ]; then
    ENV_FILE="$ROOT/runs/queue/logs/$SOURCE_TAG.env"
    if [ -f "$ENV_FILE" ]; then
        declare -A CALLER_ENV=()
        while IFS= read -r key; do
            if [ -n "${!key+x}" ]; then
                CALLER_ENV["$key"]=${!key}
            fi
        done < <(sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$ENV_FILE")

        . "$ENV_FILE"

        # 큐 sidecar의 빈 값은 설정 해제가 아니라 "미지정" 표기다.
        while IFS= read -r key; do
            if [ -z "${!key}" ]; then
                unset "$key"
            fi
        done < <(sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$ENV_FILE")

        for key in "${!CALLER_ENV[@]}"; do
            printf -v "$key" '%s' "${CALLER_ENV[$key]}"
            export "$key"
        done
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
    echo "DISPLAY가 없다. 데스크톱 터미널에서 실행하거나 LOCAL_HEADLESS=1로 검증한다." >&2
    exit 1
fi

# Git에 들어가지 않는 대용량 모션/오브젝트 데이터는 이미 받아 둔 원본 TokenHSI를
# 읽기 전용으로 공유한다. 복사하지 않고 masteer 쪽의 비어 있는 위치에만 링크한다.
if [ -n "${TOKENHSI_DATA_ROOT:-}" ]; then
    DATA_ROOT=$(realpath "$TOKENHSI_DATA_ROOT")
else
    DATA_ROOT=""
    for candidate in \
        "$ROOT/../TokenHSI/tokenhsi/data" \
        "$ROOT/TokenHSI/tokenhsi/data"; do
        if [ -d "$candidate/dataset_sit/objects" ]; then
            DATA_ROOT=$(realpath "$candidate")
            break
        fi
    done
fi

REQUIRED_DATA=(
    dataset_amass_loco/motions
    dataset_sit/motions
    dataset_sit/objects
    dataset_carry/motions
    dataset_amass_climb/motions
    dataset_amass_climb/objects
)
for relative in "${REQUIRED_DATA[@]}"; do
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
        echo "masteer 데이터 없음: $relative" >&2
        echo "TOKENHSI_DATA_ROOT=<원본 TokenHSI/tokenhsi/data>를 지정한다." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$target")"
    ln -s "$source_dir" "$target"
    echo "데이터 링크: $target -> $source_dir"
done

. "$ROOT/scripts/masteer/viz_env.sh"
viz_expand || exit 1

# 단일 GPU 로컬 PC에서는 CUDA_VISIBLE_DEVICES나 MA_GPU를 지정하지 않는다.
# 시스템이 노출한 기본 cuda:0을 그대로 쓴다.
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_TOKEN=${MA_TOKEN:-live}
export MA_SEP=${MA_SEP:-0}
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_ZERO=${MS_ZERO:-0}
export MS_SCEN=${MS_SCEN:-free}
export MS_DT=${MS_DT:-0}
export MS_DT_RAND=${MS_DT_RAND:-0}
export MS_DT_SET=${MS_DT_SET:-0,0.5,1.0,1.5,2.0}
export MS_DECEL=${MS_DECEL:-a1}
export MS_L=${MS_L:-12}
export MS_RECOV=${MS_RECOV:-1.5}
export MS_W=${MS_W:-3.0}
export MS_GAP=${MS_GAP:-1.0}
export MS_SEP=${MS_SEP:-9.0}
export MS_PLACEBO=${MS_PLACEBO:-0}
export MS_VEL_W=${MS_VEL_W:-1}
export MS_VEL_K=${MS_VEL_K:-5}
export MS_ENDCLAMP=${MS_ENDCLAMP:-0}
export MS_CLIP=${MS_CLIP:-1}
export MS_DBG=${MS_DBG:-1}
export MS_LAT_MAX=${MS_LAT_MAX:-2.2}
export MS_SCEN_CURVE=${MS_SCEN_CURVE:-0}
export MS_DRAW_SPEED=${MS_DRAW_SPEED:-1}
export MS_CAM=${MS_CAM:-top}
export MS_CAM_AGENT=${MS_CAM_AGENT:-0}
export MS_CAM_H=${MS_CAM_H:-17}
export MS_CAM_B=${MS_CAM_B:-9}
export MS_SEED=${MS_SEED:-0}

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

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "conda 초기화 파일 없음: $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
fi

# conda 초기화 스크립트는 미정의 변수를 참조할 수 있어 nounset을 사용하지 않는다.
. "$CONDA_BASE/etc/profile.d/conda.sh"
# 명시값이 없으면 호출 셸에서 이미 활성화한 환경을 그대로 사용한다.
# 비활성 셸에서 직접 실행할 때의 로컬 기본값은 학습 wrapper와 같은 tokenhsi_koo다.
VIEW_CONDA_ENV=${TOKENHSI_CONDA_ENV:-${CONDA_DEFAULT_ENV:-tokenhsi_koo}}
conda activate "$VIEW_CONDA_ENV"

SAFE_NAME=$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9_.-' '_')
CFG_SRC="$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
CFG=$(mktemp "/tmp/masteer_view_local_${SAFE_NAME}.XXXXXX.yaml")
SNAP_DIR=$(mktemp -d "/tmp/masteer_view_local_${SAFE_NAME}.XXXXXX")
mkdir -p "$SNAP_DIR/nn"
SNAP="$SNAP_DIR/nn/Humanoid.pth"
cleanup() {
    rm -f -- "$CFG"
    rm -rf -- "$SNAP_DIR"
}
trap cleanup EXIT INT TERM

sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    "$CFG_SRC" > "$CFG"

# 학습 중 저장 파일을 직접 읽지 않도록 스냅샷을 만든다.
cp -- "$CKPT" "$SNAP"

RUN_ARGS=(--test --eval_task carry)
MODE=viewer
START_MODE="test(mixed skill)"
if [ "${MS_EVAL:-0}" != 0 ]; then
    RUN_ARGS+=(--eval)
    START_MODE="final-eval(loco)"
fi
if [ "${LOCAL_HEADLESS:-0}" != 0 ]; then
    RUN_ARGS+=(--headless)
    MODE=headless
fi

echo "=============================================================="
echo " 모드       $MODE${DISPLAY:+  DISPLAY=$DISPLAY}"
echo " 정책       $CKPT"
echo " task       ${MS_TASK:-HumanoidMASteerCarry}"
echo " stage1     $BASE_CKPT"
echo " 데이터     $DATA_ROOT"
echo " 시나리오   ${MS_VIZ:-(직접 지정)}"
echo " 시작모드   $START_MODE  (agent별 독립 샘플)"
echo " env        $ENVS x 2명  배치=$MS_SCEN  CLIP=$MS_CLIP  카메라=$MS_CAM agent=$MS_CAM_AGENT"
echo " 종료       Isaac Gym 창에서 ESC 또는 터미널에서 Ctrl-C"
echo "=============================================================="

cd "$REPO"
python -u ./tokenhsi/run.py \
    --task "${MS_TASK:-HumanoidMASteerCarry}" \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$BASE_CKPT" \
    --checkpoint "$SNAP" \
    --num_envs "$ENVS" \
    --seed "$MS_SEED" \
    "${RUN_ARGS[@]}"
