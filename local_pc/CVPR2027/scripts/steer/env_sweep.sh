#!/bin/bash
# Find the largest num_envs that runs with NO PhysX overflow warning.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/steer/env_sweep.sh "4096:300: 16384:300:_buf40"
#
# Each spec is num_envs:cell_metres:cfg_suffix. A run that merely survives is
# not good enough: the warning means PhysX is dropping contacts, which is the
# state every steer run has silently been in, so warn=0 is the pass condition
# and fps is only the tiebreak.
set -u
cd /home/hwanhee/CVPR2027
OUT=${OUT:-/tmp/claude-1003/-home-hwanhee-CVPR2027/1158558b-7e6f-447b-99ef-2ec3d701110a/scratchpad}

for spec in $1; do
    IFS=: read -r E CELL SUF <<< "$spec"
    L=$OUT/sw_${E}_${CELL}${SUF}.log
    STEER_CELL=$CELL timeout 900 bash -c "
        cd /home/hwanhee/CVPR2027/TokenHSI-steer
        source /home/hwanhee/anaconda3/etc/profile.d/conda.sh; conda activate tokenhsi
        STEER_CURVATURE=0.15 python -u ./tokenhsi/run.py --task HumanoidSteerCarry \
          --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
          --cfg_env tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_steer_carry${SUF}.yaml \
          --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
          --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth --headless \
          --num_envs $E --max_iterations 14 --output_path output/steer/sw_$E" > "$L" 2>&1
    rc=$?
    printf "envs=%-6s cell=%-4s cfg=%-7s rc=%-4s warn=%s need=%-11s fps=%s\n" \
        "$E" "$CELL" "${SUF:-stock}" "$rc" \
        "$(grep -c foundLostAggregatePairs "$L")" \
        "$(grep -o 'Capacity to [0-9]*' "$L" | sort -u | head -1 | grep -oE '[0-9]+')" \
        "$(grep -oE 'fps total: [0-9.]+' "$L" | tail -1 | grep -oE '[0-9.]+$')"
done
