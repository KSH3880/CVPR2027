#!/bin/bash
# Progress of the running batch, read from the checkpoints themselves.
#   bash scripts/steer/progress.sh [tag-prefix] [target-epochs]
#
# Two things make this less obvious than tailing a log. The trainer's stdout is
# block-buffered into the queue log, so "saving checkpoint" can sit unflushed
# for many minutes while the run is perfectly healthy. And rl_games overwrites
# a single Humanoid.pth rather than numbering saves, so there is nothing to
# count on disk either. The epoch stored inside the checkpoint is exact.
cd /home/hwanhee/CVPR2027
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
python - "${1:-f1_}" "${2:-6000}" <<'PY'
import glob, os, sys, time, warnings
import torch
warnings.filterwarnings("ignore")

prefix, target = sys.argv[1], int(sys.argv[2])
now = time.time()
print(f"{'job':16s} {'epoch':>7s} {'rate/min':>9s} {'eta':>7s} {'last':>7s}")
for p in sorted(glob.glob(f"TokenHSI-steer/output/steer/{prefix}*/*/nn/Humanoid.pth")):
    job = p.split("/")[3]
    try:
        ep = int(torch.load(p, map_location="cpu").get("epoch", 0))
    except Exception as e:
        print(f"{job:16s} {'?':>7s}  (unreadable: {type(e).__name__})")
        continue
    started = os.path.getctime(os.path.dirname(os.path.dirname(p)))
    mins = max((os.path.getmtime(p) - started) / 60, 1e-6)
    rate = ep / mins
    eta = (target - ep) / rate / 60 if rate > 0 else float("nan")
    print(f"{job:16s} {ep:7d} {rate:9.1f} {eta:6.1f}h {int((now - os.path.getmtime(p)) / 60):5d}m")
PY
