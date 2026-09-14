#!/bin/bash
set -eo pipefail
cd /home/hwanhee/koo_cvpr
source runs/queue/logs/ms57_ms18e9000_sharedgoal_w2s_boot15_6000_s0.env
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi_koo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1
export STACK_PROBE_OUT=/home/hwanhee/koo_cvpr/runs/stack_bootstrap/production_obs_fix_20260914
export TORCH_EXTENSIONS_DIR=/home/hwanhee/koo_cvpr/runs/stack_bootstrap/obs_compare_20260914/torch_extensions
export MPLCONFIGDIR=$STACK_PROBE_OUT/mpl XDG_CACHE_HOME=$STACK_PROBE_OUT/cache
export STACK_BOOTSTRAP_FRAC=0 STACK_BOOTSTRAP_EVAL=0 STACK_REHEARSAL_FRAC=0
unset STACK_BOOTSTRAP_LOAD STACK_BOOTSTRAP_SAVE STACK_BOOTSTRAP_SAVE_ONCE MA_VIDEO MA_METRICS STACK_TRACE
cd /home/hwanhee/koo_cvpr/TokenHSI-masteer
python -u ../runs/stack_bootstrap/production_obs_fix_20260914/verify.py --task HumanoidMASequentialStackRelease --cfg_train "$MS_TRAINCFG" --cfg_env "/home/hwanhee/koo_cvpr/runs/stack_bootstrap/natural_obs_compare_20260914/env.yaml" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --hrl_checkpoint "$MS_CKPT" --checkpoint "$MA_INIT_CKPT" --test --eval --eval_task carry --headless --graphics_device_id -1 --seed 0
