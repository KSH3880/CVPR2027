#!/bin/bash
# F4: one steer-carry slot. F3 와 같고 envSpacing 손잡이 이름만 다르다.
#   STEER_TAG=f3_base STEER_CFG=f2_base bash scripts/steer/f2_train.sh
#
# Everything env-scaling-related is pinned here rather than left to the caller,
# because the four settings only make sense together: 4096 envs needs a 32M
# contact-pair buffer to stop PhysX dropping contacts, needs minibatch 32768 to
# keep the number of sequential updates per rollout at the 2048 reference, and
# needs spawns spread so the broadphase pair count does not blow up. See
# docs/ENV_SCALING.md.
#
# A per-batch copy, never edited while jobs are running: bash reads a script
# incrementally, so editing one under a live job truncates it mid-parse.
set -e
cd /home/hwanhee/CVPR2027/TokenHSI-steer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

TAG=${STEER_TAG:?STEER_TAG is required}
CFG=${STEER_CFG:-f2_base}
ITERS=${STEER_ITERS:-3000}
ENVS=${STEER_ENVS:-4096}
export STEER_CELL=${STEER_CELL:-60}

# STEER_RESUME=<tag> continues that run instead of starting over. rl_games calls
# agent.restore() before train(), and set_full_state_weights brings back epoch,
# frame, the Adam state and last_mean_rewards -- so 6000 -> 12000 costs the
# difference, not the whole thing. Worth having: the shipped checkpoints were
# trained far longer than anything here (stage1 40k epochs / 5.24 B frames,
# stage2 terrainShape_carry 50k / 3.28 B), and our 6000 iter is 0.79 B, about a
# quarter of stage2's samples. Every axis so far was ranked inside that window.
# !! 이 손잡이는 원래 STEER_SPACING 이었다. env 코드가 같은 이름을 **경로 창의 점
# 간격**(기본 0.4 m)으로 이미 쓰고 있어서, envSpacing=5 를 주면 창 간격까지 5 m 가 됐다.
# K=6 창이 2.4 m 가 아니라 30 m 를 보게 되어 f18/f19/f20 세 런이 통째로 무효가 됐다.
# STEER_ENVSPACING / STEER_CP 는 환경 cfg 를 건드린다. cfg_train(STEER_CFG)과는 다른
# 파일이라 손잡이가 따로 필요하다. lab A 에서 envSpacing>=5 면 접촉쌍 요구가
# 24M -> 1M 미만으로 떨어졌다 -- envSpacing 0 이면 env 원점이 전부 겹쳐 플랫폼
# 8,192 장이 한 점에 주차되기 때문이다. 주면 생성본을 만들어 그것을 쓴다.
ENV_CFG=tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry_cp32.yaml
if [ -n "${STEER_ENVSPACING:-}${STEER_CP:-}" ]; then
    GEN=/home/hwanhee/CVPR2027/runs/gen_cfgs/steer/env_${TAG}.yaml
    python3 - "$ENV_CFG" "$GEN" "${STEER_ENVSPACING:-}" "${STEER_CP:-}" <<'PY'
import re, sys
src, dst, sp, cp = sys.argv[1:5]
s = open(src).read()
if sp:
    s = re.sub(r"^(\s*)envSpacing:.*$", rf"\g<1>envSpacing: {sp}", s, count=1, flags=re.M)
if cp:
    s = re.sub(r"^(\s*)max_gpu_contact_pairs:.*$",
               rf"\g<1>max_gpu_contact_pairs: {int(float(cp) * 1024 * 1024)}", s, count=1, flags=re.M)
open(dst, "w").write(s)
PY
    ENV_CFG=$GEN
    echo "[f3_train] env cfg -> $GEN  (spacing=${STEER_ENVSPACING:-orig} cp=${STEER_CP:-orig}M)"
fi

RESUME=${STEER_RESUME:-}
CK=""
if [ -n "$RESUME" ]; then
    CK=$(ls -d output/steer/"$RESUME"/*/nn/Humanoid.pth 2>/dev/null | head -1)
    [ -n "$CK" ] || { echo "f3_train: no checkpoint for $RESUME"; exit 1; }
    echo "[f3_train] resuming from $CK -> $ITERS iter"
fi

python -u ./tokenhsi/run.py --task HumanoidSteerCarry \
    --cfg_train /home/hwanhee/CVPR2027/runs/gen_cfgs/steer/${CFG}.yaml \
    --cfg_env "$ENV_CFG" \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    ${CK:+--checkpoint "$CK"} \
    --num_envs "$ENVS" --max_iterations "$ITERS" \
    --output_path output/steer/${TAG} \
    --headless
