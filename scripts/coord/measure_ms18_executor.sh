#!/bin/bash
# Measure frozen ms18 response without training the executor or coordinator.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-coord"
TAG=${1:-ms18_executor_sysid_v1_s0}
EXEC_INPUT=${2:-ms18_maskteam_origscale_c06_s0}
ENVS=${MS_MEASURE_ENVS:-48}
STEPS=${MS_MEASURE_STEPS:-600}
SEED=${MS_MEASURE_SEED:-0}
SCENARIOS=${MS_MEASURE_SCENARIOS:-"free cross"}

if ! [[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "잘못된 tag: $TAG" >&2; exit 2
fi
if ! [[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || [ "$ENVS" -lt 6 ]; then
    echo "MS_MEASURE_ENVS는 6 이상의 정수여야 한다: $ENVS" >&2; exit 2
fi
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MS_MEASURE_STEPS는 양의 정수여야 한다: $STEPS" >&2; exit 2
fi

resolve_file() {
    local value=$1 candidate
    for candidate in "$value" "$ROOT/$value" "$REPO/$value"; do
        if [ -f "$candidate" ]; then realpath "$candidate"; return 0; fi
    done
    return 1
}

if EXEC_CKPT=$(resolve_file "$EXEC_INPUT"); then :
else
    EXEC_CKPT=$(find "$ROOT/TokenHSI-masteer/output/masteer/$EXEC_INPUT" \
        -type f -name Humanoid.pth -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [ -z "$EXEC_CKPT" ] || [ ! -f "$EXEC_CKPT" ]; then
    echo "ms18 체크포인트 없음: $EXEC_INPUT" >&2; exit 3
fi

BASE_CKPT=""
for candidate in "$ROOT/../TokenHSI/output/tokenhsi/ckpt_stage1.pth" \
                 "$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth"; do
    if [ -f "$candidate" ]; then BASE_CKPT=$(realpath "$candidate"); break; fi
done
if [ -z "$BASE_CKPT" ]; then echo "stage1 체크포인트 없음" >&2; exit 3; fi

OUT_ROOT="$ROOT/runs/coord_measurement/$TAG"
if [ -e "$OUT_ROOT" ]; then
    echo "같은 측정 tag가 이미 있다: $OUT_ROOT" >&2; exit 4
fi
mkdir -p "$OUT_ROOT" "$ROOT/runs/gen_cfgs/coord_measurement"
CFG_SRC="$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
CFG="$ROOT/runs/gen_cfgs/coord_measurement/$TAG.yaml"
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' "$CFG_SRC" > "$CFG"

if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    elif [ -f /home/cvlab/anaconda3/etc/profile.d/conda.sh ]; then CONDA_BASE=/home/cvlab/anaconda3
    else echo "conda를 찾지 못했다" >&2; exit 1; fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"

export COORD_PROVIDER=external COORD_MODEL=c2 COORD_DRAW_CANDIDATES=0
export COORD_REPLAN_STEPS=6 COORD_SPEED_LIMIT=1
export MA_TOKEN=mask MA_TOKENIZER_ZERO=1 MA_SEP=0 MA_SPAWN_GAP=1.0
export MS_MRAND=0 MS_M_LO=0.25 MS_CLIP=1 MS_REWARD_OUTER=1
export MS_POS_C=0.6 MS_VEL_W=1 MS_DBG=0
export MS_MEASURE_STEPS=$STEPS MS_MEASURE_REPLAN_STEPS=6 MS_SEED=$SEED
export MS_MEASURE_EXEC_CKPT=$EXEC_CKPT
unset MA_LAYOUT MA_LAYOUT_D MA_LAYOUT_S MA_LAYOUT_L MS_SCEN_CURVE MS_VIZ MA_C

for scenario in $SCENARIOS; do
    case "$scenario" in free|cross) ;; *) echo "잘못된 scenario: $scenario" >&2; exit 2 ;; esac
    scenario_out="$OUT_ROOT/$scenario"
    mkdir -p "$scenario_out"
    export MS_SCEN=$scenario MS_MEASURE_OUT=$scenario_out
    echo "MS18_MEASURE_START tag=$TAG scenario=$scenario envs=$ENVS steps=$STEPS"
    cd "$REPO"
    python -u -m coordinator.measure_executor \
        --test --headless --task HumanoidMACoordCarry \
        --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
        --cfg_env "$CFG" \
        --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
        --hrl_checkpoint "$BASE_CKPT" --checkpoint "$EXEC_CKPT" \
        --num_envs "$ENVS" --seed "$SEED" \
        --output_path "$scenario_out/ms18_unused" \
        > "$scenario_out/run.log" 2>&1
    rg 'MS18_MEASURE_SUMMARY' "$scenario_out/run.log"
done

echo "MS18_MEASURE_DONE tag=$TAG output=$OUT_ROOT"
