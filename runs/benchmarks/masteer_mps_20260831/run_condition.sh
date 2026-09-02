#!/bin/bash
set -uo pipefail

ROOT=/home/cvlab/Desktop/CVPR2027/CVPR2027
MODE=${1:?usage: run_condition.sh <nomps|mps> <jobs> [iters]}
JOBS=${2:?usage: run_condition.sh <nomps|mps> <jobs> [iters]}
ITERS=${3:-50}
OUT="$ROOT/runs/benchmarks/masteer_mps_20260831/${MODE}_n${JOBS}"
REPO="$ROOT/TokenHSI-masteer"
BASE="$ROOT/../TokenHSI/output/tokenhsi/ckpt_stage1.pth"
ENV_CFG="$ROOT/runs/gen_cfgs/masteer/ms14_vw1L_s0.yaml"
TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml

. /home/cvlab/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

mkdir -p "$OUT"
printf '%s\n' "$(date +%s.%N)" > "$OUT/group_start.txt"

pids=()
for ((job=0; job<JOBS; job++)); do
    job_out="$OUT/job${job}"
    mkdir -p "$job_out"
    (
        export MA_TOKENIZER_ZERO=1 MA_TOKEN=live
        export MA_SEP=0 MA_SPAWN_GAP=1.0 MA_C=0 MA_BETA=0
        export MS_MRAND=4 MS_M_LO=0.25 MS_VEL_W=1 MS_CLIP=1
        export MS_SCEN=free MS_ZERO=0 MS_SEED="$job"
        cd "$REPO"
        printf '%s\n' "$(date +%s.%N)" > "$job_out/start.txt"
        python -u ./tokenhsi/run.py \
            --task HumanoidMASteerCarry \
            --cfg_train "$TRAIN_CFG" \
            --cfg_env "$ENV_CFG" \
            --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
            --hrl_checkpoint "$BASE" \
            --num_envs 2048 --headless --seed "$job" \
            --max_iterations "$ITERS" --output_path "$job_out/output" \
            > "$job_out/run.log" 2>&1
        rc=$?
        printf '%s\n' "$rc" > "$job_out/rc.txt"
        printf '%s\n' "$(date +%s.%N)" > "$job_out/end.txt"
        exit "$rc"
    ) &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
printf '%s\n' "$(date +%s.%N)" > "$OUT/group_end.txt"
exit "$rc"
