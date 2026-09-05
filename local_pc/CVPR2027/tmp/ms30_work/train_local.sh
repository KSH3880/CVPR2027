#!/bin/bash
# 로컬 GPU 머신에서 masteer adapt 정책을 학습한다.
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
DDP_N=${MS_DDP_N:-1}
SAVE_LATEST=${MS_SAVE_LATEST:-1000}
SAVE_ARCHIVE=${MS_SAVE_ARCHIVE:-$SAVE_LATEST}

for value in "$ENVS" "$AGENTS" "$CP" "$ITERS" "$DDP_N" "$SAVE_LATEST" "$SAVE_ARCHIVE"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "env/agent/contact/iteration 값은 양의 정수여야 한다: $value" >&2
        exit 2
    fi
done
if (( ENVS % DDP_N != 0 )); then
    echo "MS_ENVS=$ENVS 는 MS_DDP_N=$DDP_N 로 나누어지지 않는다." >&2
    exit 2
fi
LOCAL_ENVS=$((ENVS / DDP_N))

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

INIT_CKPT=""
if [ -n "${MA_INIT_CKPT:-}" ]; then
    if ! INIT_CKPT=$(resolve_file "$MA_INIT_CKPT"); then
        echo "초기 adapt 체크포인트 없음: $MA_INIT_CKPT" >&2
        exit 1
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
python3 - "$CFG_SRC" "$CFG" "$AGENTS" "$CP" "$LOCAL_ENVS" "$SPACING" <<'PY'
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

if [ "$MB" != 0 ] || [ "$DDP_N" != 1 ] || [ "$SAVE_LATEST" != 1000 ] || [ "$SAVE_ARCHIVE" != "$SAVE_LATEST" ]; then
    TRAIN_LOCAL="$ROOT/runs/gen_cfgs/masteer/${TAG}_train.yaml"
    python3 - "$TRAIN_CFG" "$TRAIN_LOCAL" "$MB" "$DDP_N" "$SAVE_LATEST" "$SAVE_ARCHIVE" <<'PY'
import re
import sys
from pathlib import Path

src, dst, mb, ddp_n, latest, archive = sys.argv[1:]
text = Path(src).read_text()
if int(mb) != 0:
    text = re.sub(r"^(\s*)minibatch_size:.*$", rf"\1minibatch_size: {mb}", text, flags=re.M)
if int(ddp_n) > 1:
    for key in ("minibatch_size", "amp_minibatch_size"):
        match = re.search(rf"^(\s*){key}:\s*(\d+)\s*$", text, flags=re.M)
        if match is None:
            raise SystemExit(f"train cfg에 {key}가 없다: {src}")
        value = int(match.group(2))
        if value % int(ddp_n):
            raise SystemExit(f"{key}={value} 는 MS_DDP_N={ddp_n}로 나누어지지 않는다")
        text = re.sub(
            rf"^(\s*){key}:\s*\d+\s*$",
            rf"\g<1>{key}: {value // int(ddp_n)}",
            text,
            flags=re.M,
        )
text = re.sub(r"^(\s*)save_frequency:.*$", rf"\1save_frequency: {latest}", text, flags=re.M)
if re.search(r"^\s*save_archive_frequency:", text, flags=re.M):
    text = re.sub(r"^(\s*)save_archive_frequency:.*$",
                  rf"\1save_archive_frequency: {archive}", text, flags=re.M)
else:
    text = re.sub(r"^(\s*)(save_intermediate:.*)$",
                  rf"\1\2\n\1save_archive_frequency: {archive}", text,
                  count=1, flags=re.M)
Path(dst).write_text(text)
PY
    TRAIN_CFG="$TRAIN_LOCAL"
fi

# ms14_vw1L과 같은 기본값까지 명시해 학습과 추론에서 재현한다.
export MS_TAG="$TAG" MS_MODE=adapt
export MS_TASK=${MS_TASK:-HumanoidMASteerCarry}
export MS_TRAINCFG="$TRAIN_CFG" MS_AGENTS="$AGENTS" MS_ENVS="$ENVS"
export MS_CP="$CP" MS_SPACING="$SPACING" MS_ITERS="$ITERS" MS_SEED="$SEED" MS_DDP_N="$DDP_N"
export MS_CKPT="$BASE_CKPT"
export MA_INIT_CKPT="$INIT_CKPT"
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
    MS_MODE MS_TASK MS_TRAINCFG MS_AGENTS MS_ENVS MS_CP MS_SPACING MS_ITERS MS_SEED MS_DDP_N MS_CKPT MA_INIT_CKPT
    MS_SAVE_LATEST MS_SAVE_ARCHIVE
    MA_TOKENIZER_ZERO MA_TOKEN MA_FREEZE_NEW_CARRY MA_ADAPTER_ONLY MA_FREEZE_INPUT_RMS MA_SEP MA_SPAWN_GAP MA_C MA_BETA
    MS_MRAND MS_M_LO MS_VEL_W MS_REWARD_OUTER MS_POS_C MS_CLIP
    MS_SCEN MS_ZERO MS_GRADCHK
    REL_SEED REL_TOP_LO REL_TOP_HI REL_SUPPORT_MODE REL_RELEASE_SCALE REL_SUPPORT_XY_LO REL_SUPPORT_XY_HI REL_SUPPORT_MARGIN
    REL_TARGET_DIST_LO REL_TARGET_DIST_HI REL_TARGET_JITTER_DEG REL_CLEAR_LO REL_CLEAR_HI
    REL_CONTACT_TOL REL_COM_MARGIN REL_ENTRY_LIN REL_ENTRY_ANG REL_ENTRY_STEPS
    REL_HAND_CLEAR REL_HAND_STEPS REL_STABLE_LIN REL_STABLE_ANG REL_STABLE_STEPS REL_DROP_Z
    STACK_SEED STACK_SIZE_MARGIN STACK_GOAL_X STACK_GOAL_Y STACK_STAGE_DIST STACK_STAGE_Z STACK_STAGE_TOL
    STACK_BODY_CLEAR STACK_RETREAT_DIST STACK_RETREAT_SCALE STACK_XY_TOL STACK_Z_TOL STACK_ENTRY_LIN STACK_ENTRY_ANG
    STACK_ENTRY_STEPS STACK_HAND_CLEAR STACK_HAND_STEPS STACK_HAND_ONLY_SWITCH STACK_FOOT_CLEAR STACK_FOOT_BOX_W STACK_CARRY_FOOT_GATE STACK_CARRY_FOOT_XY STACK_CARRY_FOOT_Z STACK_STABLE_LIN STACK_STABLE_ANG STACK_TOP_STEPS
    STACK_DROP_XY STACK_TRANSITION_BONUS STACK_CLEAR_BONUS STACK_CLEAR_PROGRESS_W STACK_CLEAR_SPEED_W STACK_CLEAR_STEER_W
    STACK_CLEAR_VEL_K STACK_CLEAR_LAT_K STACK_CLEAR_MIN_SPEED STACK_CLEAR_HARD_GATE STACK_CLEAR_XY_TOL STACK_CLEAR_Z_TOL STACK_CLEAR_STABLE_LIN STACK_CLEAR_STABLE_ANG STACK_CLEAR_MOTION_GATE STACK_CLEAR_SIGNED
    STACK_REHEARSAL_FRAC STACK_VIRTUAL_RETREAT_BOX
)
{
    for name in "${REPLAY_NAMES[@]}"; do
        if [ -n "${!name+x}" ]; then
            printf 'export %s=%q\n' "$name" "${!name}"
        else
            # 미지정과 빈 문자열은 Python 설정 파서에서 의미가 다르다.
            printf 'unset %s\n' "$name"
        fi
    done
} > "$ROOT/runs/queue/logs/$TAG.env"

# Python env가 읽는 실험 노브를 명시적으로 전달한다. 새 노브를 추가하면 여기도 넣는다.
export MA_TAU MA_STILL_V MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_D MA_LAYOUT_S
export MA_K MA_TEAM MA_MKSPN MA_TREF MA_NOFREEZE MA_FREEZE_NEW_CARRY MA_ADAPTER_ONLY MA_FREEZE_INPUT_RMS MA_TRAJ MA_TRAJ_ENVS MA_TRAJ_STEPS
export MA_DHIST MA_DHIST_STEPS
export MS_K MS_M_NOM MS_POS_C MS_REWARD_OUTER MS_BACK MS_PIN MS_LAT_MAX MS_TURN_MAX MS_HUMP2
export MS_SKEW MS_SPREAD_MIN MS_SPREAD_MAX MS_LAT_FRAC MS_GAP MS_DT MS_DT_RAND
export MS_DT_SET MS_W MS_L MS_SEP MS_ENC_R MS_VEL_K MS_ENDCLAMP MS_DECEL
export MS_PLACEBO MS_RECOV MS_DBG
export REL_SEED REL_TOP_LO REL_TOP_HI REL_SUPPORT_MODE REL_RELEASE_SCALE REL_SUPPORT_XY_LO REL_SUPPORT_XY_HI REL_SUPPORT_MARGIN
export REL_TARGET_DIST_LO REL_TARGET_DIST_HI REL_TARGET_JITTER_DEG REL_CLEAR_LO REL_CLEAR_HI
export REL_CONTACT_TOL REL_COM_MARGIN REL_ENTRY_LIN REL_ENTRY_ANG REL_ENTRY_STEPS
export REL_HAND_CLEAR REL_HAND_STEPS REL_STABLE_LIN REL_STABLE_ANG REL_STABLE_STEPS REL_DROP_Z
export STACK_SEED STACK_SIZE_MARGIN STACK_GOAL_X STACK_GOAL_Y STACK_STAGE_DIST STACK_STAGE_Z STACK_STAGE_TOL
export STACK_BODY_CLEAR STACK_RETREAT_DIST STACK_RETREAT_SCALE STACK_XY_TOL STACK_Z_TOL STACK_ENTRY_LIN STACK_ENTRY_ANG
export STACK_ENTRY_STEPS STACK_HAND_CLEAR STACK_HAND_STEPS STACK_HAND_ONLY_SWITCH STACK_FOOT_CLEAR STACK_FOOT_BOX_W STACK_CARRY_FOOT_GATE STACK_CARRY_FOOT_XY STACK_CARRY_FOOT_Z STACK_STABLE_LIN STACK_STABLE_ANG STACK_TOP_STEPS
export STACK_DROP_XY STACK_TRANSITION_BONUS STACK_CLEAR_BONUS STACK_CLEAR_PROGRESS_W STACK_CLEAR_SPEED_W STACK_CLEAR_STEER_W
export STACK_CLEAR_VEL_K STACK_CLEAR_LAT_K STACK_CLEAR_MIN_SPEED STACK_CLEAR_HARD_GATE STACK_CLEAR_XY_TOL STACK_CLEAR_Z_TOL STACK_CLEAR_STABLE_LIN STACK_CLEAR_STABLE_ANG STACK_CLEAR_MOTION_GATE STACK_CLEAR_SIGNED
export STACK_REHEARSAL_FRAC STACK_VIRTUAL_RETREAT_BOX

if [ -n "${MA_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$MA_GPU"
fi

echo "=============================================================="
echo " tag        $TAG"
echo " 정책 시작  stage1 -> adapt (resume 아님)"
echo " env        총 $ENVS ($LOCAL_ENVS/GPU) x ${AGENTS}명  GPU=${DDP_N}  CP=${CP}M  spacing=$SPACING"
echo " 설정       OUTER=$MS_REWARD_OUTER POS_C=$MS_POS_C VEL_W=$MS_VEL_W MRAND=$MS_MRAND CLIP=$MS_CLIP seed=$SEED"
echo " iteration  $ITERS  grad-check=$MS_GRADCHK"
echo " 저장       latest=${SAVE_LATEST} iter, archive=${SAVE_ARCHIVE} iter"
if [ -n "$INIT_CKPT" ]; then
    echo " 이어학습   $INIT_CKPT"
fi
echo " 출력       $OUT_DIR"
echo "=============================================================="

cd "$REPO"
RUN_MAX="$ITERS"
RESUME_ARGS=()
if [ -n "$INIT_CKPT" ]; then
    BASE_EPOCH=$(python -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu').get('epoch', 0)))" "$INIT_CKPT")
    RUN_MAX=$((BASE_EPOCH + ITERS))
    RESUME_ARGS=(--checkpoint "$INIT_CKPT" --resume 1)
    echo "[masteer-local] resume epoch=$BASE_EPOCH +$ITERS -> max_iterations=$RUN_MAX"
fi
RUNNER=(python -u)
DDP_ARGS=()
if [ "$DDP_N" -gt 1 ]; then
    export PYTHONPATH="$ROOT/ddp_shim${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    RUNNER=(torchrun --standalone --nproc_per_node="$DDP_N")
    DDP_ARGS=(--horovod)
fi
"${RUNNER[@]}" ./tokenhsi/run.py \
    --task "$MS_TASK" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file "$MOTION_FILE" \
    --hrl_checkpoint "$BASE_CKPT" \
    "${RESUME_ARGS[@]}" \
    --num_envs "$LOCAL_ENVS" --headless --seed "$SEED" \
    "${DDP_ARGS[@]}" \
    --max_iterations "$RUN_MAX" --output_path "$OUT_DIR"
