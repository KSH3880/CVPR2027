#!/bin/bash
# 기존 정책으로 stack 시작 상태를 하나 확보해 저장한다. 정책은 학습하지 않는다.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?사용법: collect_stack_bootstrap.sh <tag>}
if ! [[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "잘못된 tag: $TAG" >&2
    exit 2
fi
source "$ROOT/runs/queue/logs/$TAG.env"
if [[ "${MS_TASK:-}" != HumanoidMASequentialStackRelease ]]; then
    echo "HumanoidMASequentialStackRelease 실험만 수집할 수 있다." >&2
    exit 2
fi
GPU=${MA_GPU:-7}
if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU는 6 또는 7이어야 한다: $GPU" >&2
    exit 2
fi
ENVS=${STACK_COLLECT_ENVS:-1024}
STEPS=${STACK_BOOTSTRAP_COLLECT_STEPS:-6000}
for value in "$ENVS" "$STEPS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "수집 환경 수와 스텝 수는 양의 정수여야 한다: $value" >&2
        exit 2
    fi
done
export STACK_BOOTSTRAP_SAVE=${STACK_BOOTSTRAP_SAVE:-$ROOT/runs/stack_bootstrap/$TAG.pth}
if [ -e "$STACK_BOOTSTRAP_SAVE" ]; then
    echo "시작 상태 파일이 이미 있다: $STACK_BOOTSTRAP_SAVE" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export STACK_BOOTSTRAP_SAVE_ONCE=1 STACK_BOOTSTRAP_COLLECT_STEPS=$STEPS
export STACK_BOOTSTRAP_FRAC=0.1875 STACK_BOOTSTRAP_EVAL=1
unset STACK_BOOTSTRAP_LOAD
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi_koo}"
cd "$ROOT/TokenHSI-masteer"
CKPT=$(find "output/masteer/$TAG" -name Humanoid.pth -printf '%T@ %p\n' | sort -rn | head -n 1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "체크포인트 없음: $TAG" >&2; exit 1; }
SNAP_DIR=$(mktemp -d /tmp/stack_collect.XXXXXX)
trap 'rm -rf -- "$SNAP_DIR"' EXIT
cp "$CKPT" "$SNAP_DIR/policy.pth"
CFG="$ROOT/runs/gen_cfgs/masteer/collect_${TAG}_$$.yaml"
TRAIN_CFG="$ROOT/runs/gen_cfgs/masteer/collect_${TAG}_$$_train.yaml"
python3 - "$ROOT" "$TAG" "$ENVS" "$CFG" "$TRAIN_CFG" "${MS_TRAINCFG:-}" <<'PY'
from pathlib import Path
import sys, yaml
root, tag, envs, cfg_path, train_path, train_source = sys.argv[1:]
source = Path(root) / 'runs/gen_cfgs/masteer' / (tag + '.yaml')
cfg = yaml.safe_load(source.read_text())
cfg['env']['numEnvs'] = int(envs)
cfg['env']['numAgents'] = 2
cfg['env']['enableDebugVis'] = False
Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
train = yaml.safe_load(Path(train_source).read_text())
train['params']['config']['player'] = {
    'games_num': 1000000, 'determenistic': False, 'print_stats': False,
}
Path(train_path).write_text(yaml.safe_dump(train, sort_keys=False))
PY
python -u ./tokenhsi/run.py --task "$MS_TASK" \
    --cfg_train "$TRAIN_CFG" --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "${MS_CKPT:-output/tokenhsi/ckpt_stage1.pth}" \
    --checkpoint "$SNAP_DIR/policy.pth" --test --eval_task carry \
    --headless --graphics_device_id -1 --seed "${MS_SEED:-0}"
if [ ! -f "$STACK_BOOTSTRAP_SAVE" ]; then
    echo "stack 시작 상태를 확보하지 못했습니다." >&2
    exit 1
fi
