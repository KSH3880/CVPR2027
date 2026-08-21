#!/bin/bash
# lab 슬롯 한 칸. steer 와 ma 양쪽을 같은 방식으로 잰다. (run.sh v2: 메모리 + 소수 CP)
#
#   LAB_REPO=steer|ma      어느 레포를 잴 것인가
#   LAB_ENVS               num_envs
#   LAB_CP                 max_gpu_contact_pairs, M 단위
#   LAB_SPACING            envSpacing (비우면 원본 값)
#   LAB_CELL               STEER_CELL (steer 전용, 비우면 60. "off" 면 흩뿌림 끔)
#   LAB_MB                 minibatch (비우면 원본)
#   LAB_ITERS              기본 30
#
# 요구량을 재려면 CP 를 일부러 작게 준다. 통과하면 "그 이하"까지만 알 수 있지만,
# 넘치면 로그의 `Capacity to N` 이 실제 요구량을 알려준다.
set -u
ROOT=/home/hwanhee/CVPR2027
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

TAG=${MA_TAG:?MA_TAG required}
REPO=${LAB_REPO:-ma}
ENVS=${LAB_ENVS:-4096}
CP=${LAB_CP:-4}
ITERS=${LAB_ITERS:-30}
GEN=$ROOT/runs/gen_cfgs/lab
mkdir -p "$GEN"

# 메모리 최고치를 남긴다. env 를 늘릴 때 접촉쌍보다 메모리가 먼저 막을 수 있는데,
# 안 재면 "8192 이 된다" 와 "8192 이 아슬아슬하게 된다" 를 구분할 수 없다.
# 공유 중이면 이 값은 카드 전체 사용량이다. 동거 조건은 자동 결과와 GPU 현황을 함께 본다.
MEMF=$(mktemp)
( GI=${CUDA_VISIBLE_DEVICES:-0}
  while :; do
      nvidia-smi -i "$GI" --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMF" 2>/dev/null
      sleep 10
  done ) &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null; rm -f "$MEMF"' EXIT

patch_cfg() {   # $1=src $2=dst
    python3 - "$1" "$2" "$CP" "${LAB_SPACING:-}" "${LAB_AGENTS:-}" <<'PY'
import re, sys
src, dst, cp, spacing, agents = sys.argv[1:6]
s = open(src).read()
s = re.sub(r"^(\s*)max_gpu_contact_pairs:.*$", "", s, flags=re.M)
s = re.sub(r"^(\s*)default_buffer_size_multiplier:",
           rf"\1max_gpu_contact_pairs: {int(float(cp) * 1024 * 1024)}\n\1default_buffer_size_multiplier:",
           s, count=1, flags=re.M)
if spacing:
    s = re.sub(r"^  envSpacing:.*$", f"  envSpacing: {spacing}", s, flags=re.M)
if agents:
    s = re.sub(r"^  numAgents:.*$", f"  numAgents: {agents}", s, flags=re.M)
open(dst, "w").write(s)
PY
}

if [ "$REPO" = steer ]; then
    cd $ROOT/TokenHSI-steer
    patch_cfg tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry_cp32.yaml "$GEN/$TAG.yaml"
    CELL=${LAB_CELL:-60}
    # "off" = 흩뿌림 없이 envSpacing 만으로 갈 수 있는지 본다
    [ "$CELL" = off ] && CELL=0
    STEER_CELL=$CELL STEER_PHASEFREE=1 STEER_CURVATURE=0.15 STEER_TRAJ=v3 \
    STEER_POS=latpen STEER_POS_C=0.6 STEER_K=10 \
    python -u ./tokenhsi/run.py --task HumanoidSteerCarry \
        --cfg_train $ROOT/runs/gen_cfgs/steer/f11_long.yaml \
        --cfg_env "$GEN/$TAG.yaml" \
        --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
        --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
        --num_envs "$ENVS" --max_iterations "$ITERS" \
        --output_path output/steer/_lab_$TAG --headless
else
    cd $ROOT/TokenHSI-ma
    patch_cfg tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml "$GEN/$TAG.yaml"
    TRAIN=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml
    if [ -n "${LAB_MB:-}" ]; then
        sed "s/^\( *\)minibatch_size:.*/\1minibatch_size: $LAB_MB/" "$TRAIN" > "$GEN/${TAG}_train.yaml"
        TRAIN="$GEN/${TAG}_train.yaml"
    fi
    python -u ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
        --cfg_train "$TRAIN" \
        --cfg_env "$GEN/$TAG.yaml" \
        --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
        --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
        --num_envs "$ENVS" --max_iterations "$ITERS" \
        --output_path output/ma/_lab_$TAG --headless
fi
rc=$?

LOG=$ROOT/runs/queue/logs/${TAG}.log
warn=$(grep -c foundLostAggregatePairs "$LOG" 2>/dev/null || true)
need=$(grep -o 'Capacity to [0-9]*' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
# 정상 상태 처리량: 앞쪽 7 iteration 은 기동이 섞여 버린다
sfps=$(grep -oE 'fps step: [0-9.]+' "$LOG" 2>/dev/null | grep -oE '[0-9.]+$' | tail -n +8 |
       awk '{s+=$1;c++} END{if(c)printf "%.0f", s/c}')
kill $MEMPID 2>/dev/null
memMiB=$(sort -n "$MEMF" 2>/dev/null | tail -1)
printf 'LAB tag=%s repo=%s envs=%s cp=%sM spacing=%s cell=%s rc=%s warn=%s needM=%s memMiB=%s sfps=%s\n' \
    "$TAG" "$REPO" "$ENVS" "$CP" "${LAB_SPACING:-orig}" "${LAB_CELL:-60}" "$rc" \
    "${warn:-0}" "$(awk -v n="${need:-0}" 'BEGIN{printf "%.1f", n/1048576}')" "${memMiB:-NA}" "${sfps:-NA}"
exit $rc
