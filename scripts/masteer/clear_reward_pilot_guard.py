#!/usr/bin/env python3
import os
import sys

import numpy as np


current_path = sys.argv[1]
baseline_path = sys.argv[2]
current = np.load(current_path)
baseline = np.load(baseline_path)
if current.ndim != 2 or baseline.ndim != 2:
    raise RuntimeError("metrics must be 2-D")
if current.shape[1] < 73 or baseline.shape[1] < 73:
    raise RuntimeError("pilot guard needs 73-column stack diagnostics")

pick, place, release, kind, base_disp = 41, 43, 44, 48, 49
clear_steps, retreat_arc, clear_foot = 65, 71, 72
arc_target = float(os.environ.get("PILOT_RETREAT_ARC", "0.60"))
pickup_min = float(os.environ.get("PILOT_PICKUP_MIN", "0.85"))
release_min = float(os.environ.get("PILOT_RELEASE_MIN", "0.80"))
disp_max = float(os.environ.get("PILOT_BASE_DISP_P95_MAX", "0.15"))
foot_drop_max = float(os.environ.get("PILOT_FOOT_CLEAR_DROP_MAX", "0.05"))


def rate(values):
    return float(values.mean()) if len(values) else float("nan")


def metrics(matrix):
    base = matrix[:, kind] == 1
    top = matrix[:, kind] == 0
    x = matrix[base]
    placed = x[:, place] >= 0
    released = x[:, release] >= 0
    post = x[released]
    pickup = min(
        rate(matrix[base, pick] >= 0),
        rate(matrix[top, pick] >= 0),
    )
    release_given_place = rate(released[placed])
    displacement = (
        float(np.quantile(post[:, base_disp], 0.95))
        if len(post) else float("nan")
    )
    steps = x[:, clear_steps].sum()
    foot_clear = (
        float(x[:, clear_foot].sum() / steps)
        if steps > 0 else float("nan")
    )
    retreat = rate(x[:, retreat_arc] >= arc_target)
    return pickup, release_given_place, displacement, foot_clear, retreat


pickup, release_rate, displacement, foot_clear, retreat = metrics(current)
_, _, _, base_foot_clear, base_retreat = metrics(baseline)
pickup_ok = np.isfinite(pickup) and pickup >= pickup_min
release_ok = np.isfinite(release_rate) and release_rate >= release_min
disp_ok = np.isfinite(displacement) and displacement <= disp_max
foot_ok = (
    not np.isfinite(base_foot_clear)
    or (np.isfinite(foot_clear) and foot_clear >= base_foot_clear - foot_drop_max)
)
protect = pickup_ok and release_ok and disp_ok and foot_ok
learning = np.isfinite(retreat) and retreat > base_retreat

print(
    f"PILOT_GUARD protect={'PASS' if protect else 'FAIL'} "
    f"pickup={pickup:.3f}/{pickup_min:.3f} "
    f"release_given_place={release_rate:.3f}/{release_min:.3f} "
    f"base_disp_p95={displacement:.3f}/{disp_max:.3f} "
    f"foot_clear={foot_clear:.3f} baseline={base_foot_clear:.3f} "
    f"retreat_arc{arc_target:.2f}={retreat:.3f} baseline={base_retreat:.3f} "
    f"learning={'UP' if learning else 'NO_UP'}"
)
