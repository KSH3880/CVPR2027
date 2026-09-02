#!/bin/bash
# 로컬 단일-GPU 머신에서 masteer adapt 정책을 학습한다.
# 서버용 train.sh의 /home/hwanhee 경로와 큐 데몬에 의존하지 않는다.
#
#   MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_MRAND=4 MS_CLIP=1 \
#     MS_ITERS=9000 MS_GRADCHK=1 bash scripts/masteer/train_local.sh ms17_origscale_c06_s0
#
# stdout/stderr 로그를 남기며 분리 실행하려면:
#   nohup bash scripts/masteer/train_local.sh <tag> > runs/queue/logs/<tag>.log 2>&1 &
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
TAG=${MS_TAG:-${1:-}}

if [ -z "$TAG" ]; then
    echo "사용법: train_local.sh <tag>" >&2
    exit 2
fi
if ! [[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "잘못된 tag: $TAG" >&2
    exit 2
fi

ENVS=${MS_ENVS:-2048}
AGENTS=${MS_AGENTS:-2}
CP=${MS_CP:-4}
SPACING=${MS_SPACING:-5}
ITERS=${MS_ITERS:-9000}
SEED=${MS_SEED:-0}
MB=${MS_MB:-0}

for value in "$ENVS" "$AGENTS" "$CP" "$ITERS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "env/agent/contact/iteration 값은 양의 정수여야 한다: $value" >&2
        exit 2
    fi
done

OUT_DIR="$REPO/output/masteer/$TAG"
if [ -e "$OUT_DIR" ]; then
    echo "같은 tag 출력이 이미 있다: $OUT_DIR" >&2
    echo "체크포인트를 덮어쓰지 않으므로 새 tag를 사용한다." >&2
    exit 3
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

CFG_SRC="$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
MOTION_FILE="$REPO/tokenhsi/data/dataset_loco_sit_carry_climb.yaml"
TRAIN_SRC=${MS_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml}
if [[ "$TRAIN_SRC" = /* ]]; then
    TRAIN_CFG="$TRAIN_SRC"
else
    TRAIN_CFG="$REPO/$TRAIN_SRC"
fi
for required in "$CFG_SRC" "$MOTION_FILE" "$TRAIN_CFG"; do
    if [ ! -f "$required" ]; then
        echo "필수 파일 없음: $required" >&2
        exit 1
    fi
done

# 대용량 데이터는 view_local.sh가 만든 원본 TokenHSI 링크를 그대로 사용한다.
for relative in \
    dataset_amass_loco/motions \
    dataset_sit/motions \
    dataset_sit/objects \
    dataset_carry/motions \
    dataset_amass_climb/motions \
    dataset_amass_climb/objects; do
    if [ ! -e "$REPO/tokenhsi/data/$relative" ]; then
        echo "학습 데이터 없음: $REPO/tokenhsi/data/$relative" >&2
        echo "먼저 view_local.sh로 데이터 링크를 준비하거나 TOKENHSI 데이터를 배치한다." >&2
        exit 1
    fi
done

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

CFG="$ROOT/runs/gen_cfgs/masteer/$TAG.yaml"
mkdir -p "$(dirname "$CFG")" "$ROOT/runs/results/masteer" "$ROOT/runs/queue/logs"
python3 - "$CFG_SRC" "$CFG" "$AGENTS" "$CP" "$ENVS" "$SPACING" <<'PY'
import re
import sys
from pathlib import Path

src, dst, agents, cp, envs, spacing = sys.argv[1:]
text = Path(src).read_text()
text = re.sub(r"^  numAgents:.*$", f"  numAgents: {agents}", text, flags=re.M)
text = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {envs}", text, flags=re.M)
text = re.sub(r"^  envSpacing:.*$", f"  envSpacing: {spacing}", text, flags=re.M)
text = re.sub(
    r"^(\s*)default_buffer_size_multiplier:",
    rf"\1max_gpu_contact_pairs: {int(cp) * 1024 * 1024}\n\1default_buffer_size_multiplier:",
    text,
    count=1,
    flags=re.M,
)
Path(dst).write_text(text)
PY

if [ "$MB" != 0 ]; then
    TRAIN_LOCAL="$ROOT/runs/gen_cfgs/masteer/${TAG}_train.yaml"
    sed "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/" "$TRAIN_CFG" > "$TRAIN_LOCAL"
    TRAIN_CFG="$TRAIN_LOCAL"
fi

# ms14_vw1L과 같은 기본값까지 명시해 학습과 추론에서 재현한다.
export MS_TAG="$TAG" MS_MODE=adapt
export MS_TASK=${MS_TASK:-HumanoidMASteerCarry}
export MS_TRAINCFG="$TRAIN_CFG" MS_AGENTS="$AGENTS" MS_ENVS="$ENVS"
export MS_CP="$CP" MS_SPACING="$SPACING" MS_ITERS="$ITERS" MS_SEED="$SEED"
export MS_CKPT="$BASE_CKPT"
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1} MA_TOKEN=${MA_TOKEN:-live}
export MA_SEP=${MA_SEP:-0} MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MA_C=${MA_C:-0} MA_BETA=${MA_BETA:-0}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_VEL_W=${MS_VEL_W:-1} MS_CLIP=${MS_CLIP:-1}
export MS_REWARD_OUTER=${MS_REWARD_OUTER:-2} MS_POS_C=${MS_POS_C:-2}
export MS_SCEN=${MS_SCEN:-free} MS_ZERO=${MS_ZERO:-0}
export MS_GRADCHK=${MS_GRADCHK:-1}
export MS_METRICS=${MS_METRICS:-$ROOT/runs/results/masteer/$TAG.npy}
export MA_METRICS="$MS_METRICS"

# 환경변수가 shell에만 남아 평가에서 달라지지 않도록 재생 가능한 sidecar를 남긴다.
REPLAY_NAMES=(
    MS_MODE MS_TASK MS_TRAINCFG MS_AGENTS MS_ENVS MS_CP MS_SPACING MS_ITERS MS_SEED MS_CKPT
    MA_TOKENIZER_ZERO MA_TOKEN MA_SEP MA_SPAWN_GAP MA_C MA_BETA
    MS_MRAND MS_M_LO MS_VEL_W MS_REWARD_OUTER MS_POS_C MS_CLIP
    MS_SCEN MS_ZERO MS_GRADCHK
)
{
    for name in "${REPLAY_NAMES[@]}"; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
} > "$ROOT/runs/queue/logs/$TAG.env"

# Python env가 읽는 실험 노브를 명시적으로 전달한다. 새 노브를 추가하면 여기도 넣는다.
export MA_TAU MA_STILL_V MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_D MA_LAYOUT_S
export MA_K MA_TEAM MA_MKSPN MA_TREF MA_NOFREEZE MA_TRAJ MA_TRAJ_ENVS MA_TRAJ_STEPS
export MA_DHIST MA_DHIST_STEPS
export MS_K MS_M_NOM MS_POS_C MS_REWARD_OUTER MS_BACK MS_PIN MS_LAT_MAX MS_TURN_MAX MS_HUMP2
export MS_SKEW MS_SPREAD_MIN MS_SPREAD_MAX MS_LAT_FRAC MS_GAP MS_DT MS_DT_RAND
export MS_DT_SET MS_W MS_L MS_SEP MS_ENC_R MS_VEL_K MS_ENDCLAMP MS_DECEL
export MS_PLACEBO MS_RECOV MS_DBG

if [ -n "${MA_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$MA_GPU"
fi

echo "=============================================================="
echo " tag        $TAG"
echo " 정책 시작  stage1 -> adapt (resume 아님)"
echo " env        $ENVS x ${AGENTS}명  CP=${CP}M  spacing=$SPACING"
echo " 설정       OUTER=$MS_REWARD_OUTER POS_C=$MS_POS_C VEL_W=$MS_VEL_W MRAND=$MS_MRAND CLIP=$MS_CLIP seed=$SEED"
echo " iteration  $ITERS  grad-check=$MS_GRADCHK"
echo " 출력       $OUT_DIR"
echo "=============================================================="

cd "$REPO"
python -u ./tokenhsi/run.py \
    --task "$MS_TASK" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file "$MOTION_FILE" \
    --hrl_checkpoint "$BASE_CKPT" \
    --num_envs "$ENVS" --headless --seed "$SEED" \
    --max_iterations "$ITERS" --output_path "$OUT_DIR"
