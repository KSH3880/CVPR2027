#!/bin/bash
# ma 트랙의 한 슬롯. autofill 이 큐 표의 환경변수와 함께 이걸 부른다.
#
#   MA_MODE=sweep   짧게 돌려 PhysX 요구량과 fps 만 본다 (격자)
#   MA_MODE=test    체크포인트로 평가만 (학습 없음)
#   MA_MODE=train   stage1 에서 전체 fine-tune (평범한 빌더). --resume 필요
#   MA_MODE=adapt   adapt 스캐폴드 (backbone 동결 + 학습 토큰). --hrl_checkpoint
#
#   MA_TASK  태스크 클래스. 토큰 실험은 HumanoidMACarry
#   MA_TOKEN live|zero  teammate 토큰 값. 크기는 같고 값만 0 이 된다
#
# 통과 조건은 rc=0 이 아니라 warn=0 이다. 경고가 뜨는 설정이 더 빠른데, 그건 접촉을
# 버려서 빠른 것이다 (docs/ENV_SCALING.md).
#
# 이 파일을 고칠 때는 새로 쓰고 mv 로 갈아끼운다. 실행 중인 bash 는 스크립트를 증분으로
# 읽으므로 제자리 덮어쓰기는 돌고 있는 런을 죽인다 (전례: unexpected EOF 로 6 런 사망).
set -u
cd /home/hwanhee/CVPR2027/TokenHSI-ma
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

TAG=${MA_TAG:?MA_TAG is required}
MODE=${MA_MODE:-sweep}
ENVS=${MA_ENVS:-2048}
AGENTS=${MA_AGENTS:-1}
CP=${MA_CP:-8}                 # max_gpu_contact_pairs, M 단위 (8 = 8M)
ITERS=${MA_ITERS:-14}
SEED=${MA_SEED:-0}
MB=${MA_MB:-0}                 # minibatch. 0 이면 원본 유지.
                               # num_envs 를 바꾸면 이것도 같은 배수로 움직여야
                               # 롤아웃당 업데이트 수가 유지된다 (ENV_SCALING §1)
CKPT=${MA_CKPT:-output/tokenhsi/ckpt_stage1.pth}
# 지표는 env 안에서 스텝마다 누적하고 주기적으로 여기에 쓴다 (사후 계산 금지, PITFALLS #6)
export MA_METRICS=${MA_METRICS:-/home/hwanhee/CVPR2027/runs/results/ma/${TAG}.npy}
export MA_TAU MA_STILL_V MA_LAYOUT MA_LAYOUT_R MA_LAYOUT_L MA_LAYOUT_S MA_LAYOUT_D MA_SEP
export MA_C MA_BETA MA_K MA_TOKEN MA_TOKENIZER_ZERO MA_TEAM MA_MKSPN MA_TREF MA_NOFREEZE MA_SPAWN_GAP MA_RESUME
mkdir -p "$(dirname "$MA_METRICS")"

CFG_SRC=tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml
CFG=/home/hwanhee/CVPR2027/runs/gen_cfgs/ma/${TAG}.yaml
mkdir -p "$(dirname "$CFG")"

# 슬롯마다 자기 cfg 사본을 만든다. 돌고 있는 파일을 고치면 다른 슬롯이 같이 바뀐다.
python3 - "$CFG_SRC" "$CFG" "$AGENTS" "$CP" "$ENVS" ${MA_SPACING:+"$MA_SPACING"} <<'PY'
import re, sys
src, dst, agents, cp, envs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
s = open(src).read()
s = re.sub(r"^  numAgents:.*$", f"  numAgents: {agents}", s, flags=re.M)
# rl_games 의 num_actors 는 여기서 온다. --num_envs 로 덮어도 따라오지 않는다.
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {envs}", s, flags=re.M)
if len(sys.argv) > 6:
    s = re.sub(r"^  envSpacing:.*$", f"  envSpacing: {sys.argv[6]}", s, flags=re.M)
s = re.sub(r"^(\s*)default_buffer_size_multiplier:",
           rf"\1max_gpu_contact_pairs: {cp * 1024 * 1024}\n\1default_buffer_size_multiplier:",
           s, count=1, flags=re.M)
open(dst, "w").write(s)
PY

# adapt 는 별도 학습 설정을 쓴다 (동결·adapt mlp 설정이 거기 있다)
TRAIN_SRC=${MA_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml}
TRAIN_CFG=$TRAIN_SRC
if [ "$MB" != "0" ]; then
    TRAIN_CFG=/home/hwanhee/CVPR2027/runs/gen_cfgs/ma/${TAG}_train.yaml
    sed "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/" "$TRAIN_SRC" > "$TRAIN_CFG"
fi

COMMON=(--task "${MA_TASK:-HumanoidTrajSitCarryClimb}"
        --cfg_train "$TRAIN_CFG"
        --cfg_env "$CFG"
        --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml
        --num_envs "$ENVS" --headless --seed "$SEED")

case "$MODE" in
sweep)  python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" \
            --max_iterations "$ITERS" --output_path "output/ma/$TAG" ;;
test)   python -u ./tokenhsi/run.py "${COMMON[@]}" --checkpoint "$CKPT" \
            --test --eval --eval_task "${MA_EVAL_TASK:-carry}" ;;
# stage1 에서 이어 학습하려면 --resume 1 과 --checkpoint 가 **둘 다** 필요하다.
# --resume 가 load_checkpoint 를 켜고 --checkpoint 가 경로를 준다 (utils/config.py:125-129).
# --checkpoint 만 주면 복원이 조용히 건너뛰어져 **랜덤 초기화로 학습한다** (실측: 1000 iter -> 0.0).
# --hrl_checkpoint 는 stage2 합성 태스크가 stage1 을 별도 low-level 정책으로 쓸 때의 것이라 여기선 틀리다.
train)
        # --resume 는 epoch 카운터까지 복원한다. stage1 은 40,000 epoch 이라
        # --max_iterations 를 그대로 주면 "MAX EPOCHS NUM!" 으로 즉시 끝난다
        # (실측: 6000 을 줬더니 1 iter 만에 종료). MA_ITERS 는 "여기서 몇 번 더" 다.
        base=$(python3 -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu').get('epoch',0))" "$CKPT")
        echo "[ma] resume epoch=$base  +${ITERS}  -> max_iterations $((base + ITERS))"
        python -u ./tokenhsi/run.py "${COMMON[@]}" --checkpoint "$CKPT" --resume 1 \
            --max_iterations "$((base + ITERS))" --output_path "output/ma/$TAG" ;;
adapt)
        # adapt 는 --hrl_checkpoint 로 stage1 을 **동결 부분에** 싣는다. rl_games 의
        # epoch 카운터를 복원하지 않으므로 MA_ITERS 가 곧 실제 학습량이다
        # (train 모드처럼 40,000 을 더할 필요가 없다).
        #
        # MA_RESUME=1 : 죽은 런을 이어서 돌린다. --hrl_checkpoint(동결부, stage1) 는
        # 그대로 두고 --checkpoint 로 **자기 학습본**을 얹은 뒤 --resume 으로 epoch 을
        # 복원한다. 그러면 1900 에서 죽은 런이 MA_ITERS(3000)까지 이어진다.
        RES=""
        if [ "${MA_RESUME:-0}" = "1" ]; then
            # 가장 최근 것을 고른다. 같은 태그를 여러 번 돌리면 디렉터리가 여러 개 생기는데
            # head -1 은 순서가 보장되지 않아 옛 체크포인트를 잡을 수 있다.
            OWN=$(find "output/ma/$TAG" -name Humanoid.pth -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-)
            if [ -n "$OWN" ]; then
                cp "$OWN" "/tmp/resume_$TAG.pth"
                RES="--checkpoint /tmp/resume_$TAG.pth --resume 1"
                echo "[ma] resume $TAG <- $OWN"
            else
                echo "[ma] resume 요청됐지만 체크포인트 없음 -> 처음부터"
            fi
        fi
        python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" $RES \
            --max_iterations "$ITERS" --output_path "output/ma/$TAG" ;;
atest)  python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" \
            --test --eval --eval_task "${MA_EVAL_TASK:-carry}" ;;
*)      echo "unknown MA_MODE=$MODE"; exit 1 ;;
esac
rc=$?

# grep -c 는 0 건일 때도 "0" 을 찍고 종료코드 1 을 낸다. `|| echo 0` 을 붙이면 "0" 이
# 두 번 나와 값에 개행이 섞이고 뒤 필드가 통째로 밀린다. || true 여야 한다.
LOG=/home/hwanhee/CVPR2027/runs/queue/logs/${TAG}.log
warn=$(grep -c foundLostAggregatePairs "$LOG" 2>/dev/null || true)
need=$(grep -o 'Capacity to [0-9]*' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
fps=$(grep -oE 'fps total: [0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
# 평가는 0 회차를 버린다 -- 리셋이 안 가라앉은 상태라 값이 다르다 (PITFALLS #5)
sr=$(grep -oE "'success_rate': [0-9.]+" "$LOG" 2>/dev/null | grep -oE '[0-9.]+' |
     awk 'NR>1{s+=$1;n++} END{if(n)printf "%.4f",s/n}')
# 정상 상태 처리량: 앞쪽 절반을 버린다 (기동·워밍업이 섞인다)
sfps=$(grep -oE 'fps step: [0-9.]+' "$LOG" | grep -oE '[0-9.]+$' |
       awk '{a[NR]=$1} END{n=asort(a); s=0; c=0; for(i=int(NR/2);i<=NR;i++){s+=a[i];c++} if(c)printf "%.0f", s/c}' 2>/dev/null ||
       grep -oE 'fps step: [0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | tail -n +8 |
       awk '{s+=$1;c++} END{if(c)printf "%.0f", s/c}')
met=$(python3 - "$MA_METRICS" 2>/dev/null <<'PYX'
import sys, numpy as np
try:
    m = np.load(sys.argv[1])
except Exception:
    sys.exit()
f = m[:, 1]
ok = f >= 0
print(f"eps={m.shape[0]} fin={ok.mean():.3f} "
      f"t50={np.median(f[ok]) if ok.any() else -1:.0f} "
      f"colEp={(m[:,2]>0).mean():.3f} colStep={m[:,2].mean():.1f} "
      f"path={np.median(m[:,5]):.2f}")
PYX
)
printf 'MA_METRIC tag=%s %s\n' "$TAG" "${met:-none}"
printf 'MA_SUMMARY tag=%s agents=%s envs=%s cp=%sM mb=%s seed=%s rc=%s warn=%s need=%s fps=%s sfps=%s sr=%s\n' \
    "$TAG" "$AGENTS" "$ENVS" "$CP" "$MB" "$SEED" "$rc" "${warn:-0}" "${need:-0}" "${fps:-NA}" "${sfps:-NA}" "${sr:-NA}"
exit $rc
