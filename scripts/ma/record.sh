#!/bin/bash
# 돌고 있는 학습의 **현재 정책**을 영상으로 뽑는다.
#
#   bash scripts/ma/record.sh t1_live_s0            30초 (900프레임), env 3개
#   bash scripts/ma/record.sh t2_c05_b25 15 2       15초, env 2개
#   MA_SEP=20 bash scripts/ma/record.sh t1_live_s0  간섭 없는 배치로
#   MA_VIDEO_DIST=4 bash …                          카메라 거리 고정 (기본: 두 사람 간격에 맞춰 자동)
#
# 결과: runs/results/video/<tag>_<시각>.mp4
#
# **X11 도 VNC 도 안 쓴다.** 이 서버는 /dev/dri 권한이 없어 OpenGL 뷰어가 못 뜨고
# VNC 서버도 없다(sudo 불가). IsaacGym 카메라 센서는 X 를 안 거치고 GPU 그래픽스
# 파이프라인으로 직접 렌더하므로 이게 유일한 headless 녹화 경로다.
#
# 학습 프로세스에 붙는 게 아니라, 100 iter 마다 덮어쓰이는 체크포인트를 **복사해서**
# 별도로 띄운다. 복사 이유: 학습이 쓰는 중인 파일을 열면 반쯤 쓰인 상태를 읽는다.
set -u
TAG=${1:?사용법: record.sh <tag> [초] [env수]}
SEC=${2:-30}
N=${3:-3}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=${MA_GPU:-6}
VIEW_DRI=$(python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$GPU") || exit $?
mkdir -p "$ROOT/runs/results/video"
MP4=$ROOT/runs/results/video/${TAG}_$(date +%H%M%S).mp4

cd $ROOT/TokenHSI-ma
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

SRC=$(find output/ma/$TAG -name Humanoid.pth 2>/dev/null | head -1)
[ -z "$SRC" ] && { echo "체크포인트 없음: output/ma/$TAG"; exit 1; }
SNAP=/tmp/ma_rec_$TAG.pth; cp "$SRC" "$SNAP"
ITER=$(grep -c "fps step" $ROOT/runs/queue/logs/$TAG.log 2>/dev/null || echo "?")
echo "[rec] $TAG  iter≈$ITER  ($(( $(date +%s) - $(stat -c %Y "$SRC") ))초 전 저장본)  env $N  ${SEC}초"

CFG=$ROOT/runs/gen_cfgs/ma/rec_$TAG.yaml
python3 - "$N" "$CFG" <<'PY'
import re, sys, pathlib
s = pathlib.Path("tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml").read_text()
s = re.sub(r"^  numAgents:.*$", "  numAgents: 2", s, flags=re.M)
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[1]}", s, flags=re.M)
pathlib.Path(sys.argv[2]).write_text(s)
PY

MA_VIDEO="$MP4" MA_VIDEO_FRAMES=$((SEC * 30)) MA_VIDEO_EVERY=${MA_VIDEO_EVERY:-2} \
MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1} MA_TOKEN=${MA_TOKEN:-live} MA_SEP=${MA_SEP:-0} \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=$VIEW_DRI \
python -u ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
  --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth --checkpoint "$SNAP" \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --test --headless --eval_task carry 2>&1 | grep -E "^\[ma\] video|Traceback"

[ -s "$MP4" ] && echo "[rec] $MP4  ($(du -h "$MP4" | cut -f1))" || echo "[rec] 실패"
