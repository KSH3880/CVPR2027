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
#   MA_GPU=<물리 compute GPU index> (기본 0)
#   VIEW_GRAPHICS_GPU=<물리 graphics GPU index> (기본 MA_GPU와 같음)
#   VIEW_ENV_OVERRIDE_KEYS="MS_SCEN MS_DT"  sidecar보다 우선할 키를 명시
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
# 현재 셸에는 이전 태그를 source한 값이 남을 수 있으므로 sidecar를 기본 권위로 삼는다.
# 의도적인 실험 override만 VIEW_ENV_OVERRIDE_KEYS에 키 이름을 명시해 보존한다.
if [ -n "$SOURCE_TAG" ]; then
    ENV_FILE="$ROOT/runs/queue/logs/$SOURCE_TAG.env"
    if [ -f "$ENV_FILE" ]; then
        declare -A CALLER_ENV=()
        override_keys=${VIEW_ENV_OVERRIDE_KEYS//,/ }
        for key in $override_keys; do
            if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
                echo "VIEW_ENV_OVERRIDE_KEYS에 잘못된 키가 있다: $key" >&2
                exit 2
            fi
            if [ -n "${!key+x}" ]; then
                CALLER_ENV["$key"]=${!key}
            fi
        done

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
if [ "${LOCAL_HEADLESS:-0}" = 0 ] && command -v xdpyinfo >/dev/null 2>&1; then
    if ! xdpyinfo -display "$DISPLAY" 2>/dev/null | grep -w DRI3 >/dev/null; then
        echo "현재 DISPLAY=$DISPLAY에 DRI3 확장이 없다." >&2
        echo "Isaac Gym Vulkan viewer는 이 XWayland 세션에서 실행할 수 없다." >&2
        echo "로그아웃 후 로그인 화면의 톱니바퀴에서 'Ubuntu on Xorg'를 선택한다." >&2
        exit 2
    fi
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

# 선택한 graphics GPU의 실제 Vulkan index를 UUID로 찾는다. compute와 graphics GPU가
# 같으면 CUDA를 logical 0으로 마스킹하고, 다르면 물리 CUDA index를 그대로 사용한다.
GPU=${MA_GPU:-0}
GRAPHICS_GPU=${VIEW_GRAPHICS_GPU:-$GPU}
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "MA_GPU는 음이 아닌 정수여야 한다: $GPU" >&2
    exit 2
fi
for selected_gpu in "$GPU" "$GRAPHICS_GPU"; do
    if ! [[ "$selected_gpu" =~ ^[0-9]+$ ]] || \
       ! nvidia-smi -i "$selected_gpu" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
        echo "물리 GPU $selected_gpu를 nvidia-smi에서 찾을 수 없다" >&2
        exit 2
    fi
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$GPU" = "$GRAPHICS_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
    SIM_DEVICE=cuda:0
else
    unset CUDA_VISIBLE_DEVICES
    SIM_DEVICE=cuda:$GPU
fi
export TOKENHSI_ALLOWED_GPUS=$GRAPHICS_GPU
export TOKENHSI_VULKAN_ALLOW_EXTRA=1
export TOKENHSI_VULKAN_OUTPUT_INDEX=1
# GPU-backed Xorg에서는 NVIDIA/Isaac Gym이 고른 Vulkan 장치 순서를 그대로 써야 한다.
# DRI_PRIME 또는 단일 ICD 강제는 이 로컬 드라이버에서 draw_viewer 세그폴트를 낸다.
USE_SYSTEM_VULKAN=${TOKENHSI_USE_SYSTEM_VULKAN:-0}
if [ "$USE_SYSTEM_VULKAN" = 1 ]; then
    unset VK_ICD_FILENAMES VK_INSTANCE_LAYERS
    export TOKENHSI_DRI_PRIME=none
elif [ -z "${VK_ICD_FILENAMES:-}" ] && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
if [ -z "${TOKENHSI_DRI_PRIME:-}" ]; then
    GRAPHICS_BUS=$(nvidia-smi -i "$GRAPHICS_GPU" --query-gpu=pci.bus_id \
        --format=csv,noheader | tr '[:upper:]' '[:lower:]')
    GRAPHICS_BUS=${GRAPHICS_BUS#0000}
    export TOKENHSI_DRI_PRIME="pci-${GRAPHICS_BUS//[:.]/_}!"
fi
GRAPHICS_DEVICE_ID=$(python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$GRAPHICS_GPU")
if [ "$USE_SYSTEM_VULKAN" = 1 ]; then
    unset DRI_PRIME
else
    export DRI_PRIME=$TOKENHSI_DRI_PRIME
fi
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
echo " GPU        compute=$GPU ($SIM_DEVICE)  graphics=$GRAPHICS_GPU (Vulkan index $GRAPHICS_DEVICE_ID)"
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
    --sim_device "$SIM_DEVICE" \
    --graphics_device_id "$GRAPHICS_DEVICE_ID" \
    --rl_device "$SIM_DEVICE" \
    --seed "$MS_SEED" \
    "${RUN_ARGS[@]}"
