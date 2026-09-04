#!/usr/bin/env python3
import re
import sys

import numpy as np


tag, log_path, metrics_path = sys.argv[1:4]
agents = int(sys.argv[4])
text = open(log_path, errors="ignore").read()
m = np.load(metrics_path)

trials = [int(v) for v in re.findall(r"'num_trials': ([0-9]+)", text)]
if not trials or len(m) != sum(trials):
    raise RuntimeError(
        f"metrics flush mismatch: rows={len(m)} expected={sum(trials) if trials else 'missing'}")
if m.shape[1] < 47:
    raise RuntimeError(f"strict pickup metrics need 47 columns, got {m.shape[1]}")

agent = m[:, 0].astype(int) % agents
finished = m[:, 1] >= 0
picked = m[:, 40] > 0
close = m[:, 42] > 0
steps = np.maximum(m[:, 30], 1)
parts = [
    f"PICKUP_DIAG tag={tag}",
    f"eps={len(m)}",
    f"fin={finished.mean():.4f}",
    f"pick={picked.mean():.4f}",
    f"close={close.mean():.4f}",
    f"closeOnly={(close & ~picked).mean():.4f}",
    f"heldFrac={np.median(m[:, 41] / steps):.4f}",
    f"maxLift50={np.median(m[:, 45]):.3f}",
    f"streak50={np.median(m[:, 46]):.0f}",
    "pickA=" + ",".join(f"{picked[agent == a].mean():.4f}" for a in range(agents)),
]
if agents == 2 and len(m) % 2 == 0:
    pair_agent = agent.reshape(-1, 2)
    if np.all(pair_agent == np.arange(2)):
        parts.append(f"bothPick={np.all(picked.reshape(-1, 2), axis=1).mean():.4f}")
print(" ".join(parts))
