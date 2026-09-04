#!/bin/bash
# 시나리오 하나를 영상으로 굽는다. VNC 없이 Xvfb 위에서만 돌리고 ffmpeg 로 잡는다.
#
#   OUT=1_straight DUR=40 MS_SCEN=parallel MS_GAP=2.0 \
#     bash scripts/masteer/record.sh ms4_m4_s0
#
#   VIDEO_DIR  기본 runs/results/video/0821
#   OUT        파일 이름 (확장자 제외)
#   DUR        녹화 길이(초). 시뮬 기동 뒤부터 잰다
#   DELAY      기동 대기 상한(초). [scen] 이 찍히면 바로 진행한다
#   SETTLE     그 뒤 추가 대기(초), 기본 8
#   ENVS       기본 1 -- 한 env 만 보이게
#
# IsaacGym 왼쪽 323px 은 설정 패널이라 잘라낸다.
set -u
ROOT=/home/hwanhee/CVPR2027
VNC_DIR=/home/hwanhee/opt/vnc
TAG=${1:?사용법: record.sh <tag>}
ENVS=${ENVS:-1}
# 명령행 override가 없으면 학습 때 저장한 teammate attention 모드를 재생한다.
# 전체 sidecar는 아래의 영상 시나리오 설정을 덮을 수 있어 MA_TOKEN만 읽는다.
if [ -z "${MA_TOKEN+x}" ] && [ -f "$ROOT/runs/queue/logs/$TAG.env" ]; then
    MA_TOKEN=$(bash -c 'source "$1"; printf "%s" "${MA_TOKEN:-live}"' _ \
        "$ROOT/runs/queue/logs/$TAG.env")
    export MA_TOKEN
fi
# MS_VIZ 를 구체 변수로 편다. view.sh 와 **같은 파일**을 쓴다 -- 매핑이 갈라지면
# 뷰어에서 확인한 것과 영상이 달라진다.
. "$ROOT/scripts/masteer/viz_env.sh"
viz_expand || exit 1
# **파일명은 MS_VIZ 값 그대로다.** 그래서 영상 이름이 곧 재현 명령이 된다.
OUT=${OUT:-${MS_VIZ:-$TAG}}
DUR=${DUR:-40}
DELAY=${DELAY:-95}
VIDEO_DIR=${VIDEO_DIR:-$ROOT/runs/results/video/0821}
GPU=${MA_GPU:-4}

# 디스플레이는 파일 이름 해시로 잡아 동시 실행이 안 겹치게 한다
DISPNUM=$(( 40 + ($(echo "$OUT" | cksum | cut -d' ' -f1) % 40) ))
DISP=":$DISPNUM"
[ -e /tmp/.X11-unix/X$DISPNUM ] && DISPNUM=$((DISPNUM+40)) && DISP=":$DISPNUM"

cd $ROOT/TokenHSI-masteer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

CKPT=$(ls -t output/masteer/$TAG/*/nn/Humanoid.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "체크포인트 없음: $TAG"; exit 1; }
SNAP_DIR=$(mktemp -d "/tmp/masteer_record.XXXXXX")
mkdir -p "$SNAP_DIR/nn"
SNAP="$SNAP_DIR/nn/Humanoid.pth"
cp "$CKPT" "$SNAP"

mkdir -p "$VIDEO_DIR"
VIDEO=$VIDEO_DIR/${OUT}.mp4
CFG=$ROOT/runs/gen_cfgs/masteer/rec_${OUT}.yaml
python3 - "$ENVS" "$CFG" <<'PY'
import re, sys, pathlib
s = pathlib.Path("tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml").read_text()
s = re.sub(r"^  numAgents:.*$", "  numAgents: 2", s, flags=re.M)
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[1]}", s, flags=re.M)
# **경로가 12 m 인데 envSpacing 이 5 면 env 들이 겹쳐 보인다.** 영상에서는 치명적이다.
import os as _o
s = re.sub(r"^  envSpacing:.*$", f"  envSpacing: {_o.environ.get('REC_SPACING','15')}", s, flags=re.M)
pathlib.Path(sys.argv[2]).write_text(s)
PY

cleanup() {
    [ -n "${FFMPEG_PID:-}" ] && { kill -INT $FFMPEG_PID 2>/dev/null; wait $FFMPEG_PID 2>/dev/null; }
    kill ${SIM_PID:-} ${XVFB_PID:-} 2>/dev/null
    rm -rf -- "$SNAP_DIR"
}
trap cleanup EXIT INT TERM

Xvfb $DISP -screen 0 1600x900x24 -nolisten tcp & XVFB_PID=$!
sleep 2

DISPLAY=$DISP \
MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1} MA_TOKEN=${MA_TOKEN:-live} MA_SEP=${MA_SEP:-0} \
MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0} MS_MRAND=${MS_MRAND:-4} MS_ZERO=0 \
MS_SCEN=${MS_SCEN:-free} MS_DT=${MS_DT:-0} MS_DT_RAND=${MS_DT_RAND:-0} \
MS_DECEL=${MS_DECEL:-a1} MS_L=${MS_L:-12} MS_RECOV=${MS_RECOV:-1.5} \
MS_GAP=${MS_GAP:-1.0} MS_SEP=${MS_SEP:-9.0} \
MS_LAT_MAX=${MS_LAT_MAX:-2.2} MS_SCEN_CURVE=${MS_SCEN_CURVE:-0} \
MS_VEL_W=${MS_VEL_W:-1} MS_CLIP=${MS_CLIP:-1} MS_DRAW_SPEED=${MS_DRAW_SPEED:-1} \
MS_DBG=1 MS_SEED=${MS_SEED:-0} \
MS_CAM=${MS_CAM:-top} MS_CAM_H=${MS_CAM_H:-17} MS_CAM_B=${MS_CAM_B:-9} \
MA_TRAJ=${TRAJ_OUT:-/tmp/rec_traj_${OUT}.npy} MA_TRAJ_ENVS=1 MA_TRAJ_STEPS=900 \
CUDA_VISIBLE_DEVICES=$GPU \
python -u ./tokenhsi/run.py --task HumanoidMASteerCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth --checkpoint "$SNAP" \
    --num_envs "$ENVS" \
    --test --eval --eval_task carry --seed ${MS_SEED:-0} > /tmp/rec_${OUT}.log 2>&1 &
SIM_PID=$!

# **고정 대기는 못 믿는다.** 모션 로딩 시간이 그때그때 다르고, 짧으면 검은 화면을 찍는다.
# 자기검사(`[scen]`)가 찍히면 리셋까지 갔다는 뜻이라 그걸 기다린다.
READY=0
for _ in $(seq 1 "$DELAY"); do
    # free 모드는 자기검사를 안 돌려 [scen] 이 안 나온다. disc_reward 는 스텝마다
    # 찍히므로 둘 중 아무거나 나오면 시뮬이 돌기 시작한 것이다.
    grep -qE "\[scen\]|disc_reward" /tmp/rec_${OUT}.log 2>/dev/null && { READY=1; break; }
    kill -0 $SIM_PID 2>/dev/null || { echo "시뮬이 죽었다"; tail -5 /tmp/rec_${OUT}.log; exit 1; }
    sleep 1
done
[ "$READY" = 0 ] && { echo "기동 대기 초과 (${DELAY}s)"; tail -3 /tmp/rec_${OUT}.log; exit 1; }
sleep ${SETTLE:-8}      # 첫 프레임이 그려질 때까지

ffmpeg -nostdin -loglevel error -y -f x11grab -draw_mouse 0 -framerate 30 \
    -video_size 1276x900 -i $DISP+324,0 -t $DUR \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$VIDEO" & FFMPEG_PID=$!
wait $FFMPEG_PID
FFMPEG_PID=

echo "저장: $VIDEO  ($(du -h "$VIDEO" 2>/dev/null | cut -f1))"
# **두 사람이 실제로 움직였는지 확인한다.** env 하나만 찍으므로 한쪽이 넘어지면
# 영상 전체가 못 쓰게 되는데, 파일 크기만 봐서는 알 수 없다.
python3 - "/tmp/rec_traj_${OUT}.npy" 2>/dev/null <<'PYX'
import sys, numpy as np, os
p = sys.argv[1]
if not os.path.exists(p):
    print("  (궤적 없음 -- 검증 못 함)"); raise SystemExit
m = np.load(p)                       # (T, 1, A, 7): step, human_xy, box_xy, tar_xy
pr = m[:, 0, 0, 0]
d = np.where(np.diff(pr) < 0)[0]
b = np.concatenate([[0], d + 1, [len(pr)]])
s0, s1 = max(((b[i], b[i+1]) for i in range(len(b)-1)), key=lambda x: x[1]-x[0])
out = []
for a in range(m.shape[2]):
    h = m[s0:s1, 0, a, 1:3]
    out.append(float(np.linalg.norm(np.diff(h, axis=0), axis=-1).sum()))
flag = "" if min(out) > 2.0 else "   <-- 한쪽이 거의 안 움직였다"
print(f"  이동거리  a0 {out[0]:.1f} m   a1 {out[1]:.1f} m{flag}")
PYX
grep -E "^\[scen\]" /tmp/rec_${OUT}.log | head -1
