#!/usr/bin/env python3
import sys

import numpy as np


path = sys.argv[1]
m = np.load(path)
if m.ndim != 2 or m.shape[1] < 49:
    raise RuntimeError(f"stack stage metrics need at least 49 columns, got {m.shape}")

near, pick, broken, place, release, clear, failed, max_lift, kind = range(40, 49)
base_disp = 49 if m.shape[1] >= 50 else None


def rate(mask):
    return float(mask.mean()) if len(mask) else float("nan")


def conditional_rate(event, condition):
    return rate(event[condition])


def pickup_rate(matrix, role):
    rows = matrix[:, kind] == role
    return rate(matrix[rows, pick] >= 0)


def line(name, rows):
    x = m[rows]
    return (
        f"{name} n={len(x)} near={rate(x[:, near] >= 0):.3f} "
        f"pick={rate(x[:, pick] >= 0):.3f} "
        f"break={rate(x[:, broken] >= 0):.3f} "
        f"delivered={rate(x[:, 1] >= 0):.3f} "
        f"maxLift50={np.median(x[:, max_lift]) if len(x) else float('nan'):.3f}"
    )


base = m[:, kind] == 1
top = m[:, kind] == 0
rehearsal = m[:, kind] == -1
print(f"STACK_STAGE_DIAG file={path} rows={len(m)}")
print(line("rehearsal", rehearsal))
print(line("base", base))
print(line("top", top))

x = m[base]
if len(x):
    buckets = {
        "approach": x[:, near] < 0,
        "lift": (x[:, near] >= 0) & (x[:, pick] < 0),
        "carry/place": (x[:, pick] >= 0) & (x[:, place] < 0),
        "release": (x[:, place] >= 0) & (x[:, release] < 0),
        "clear": (x[:, release] >= 0) & (x[:, clear] < 0),
        "passed": x[:, clear] >= 0,
    }
    print("base_first_stop " + " ".join(
        f"{name}={rate(mask):.3f}" for name, mask in buckets.items()
    ))
    print(
        f"base_events place={rate(x[:, place] >= 0):.3f} "
        f"release={rate(x[:, release] >= 0):.3f} "
        f"clear={rate(x[:, clear] >= 0):.3f} "
        f"failed={rate(x[:, failed] >= 0):.3f}"
    )
    placed = x[:, place] >= 0
    released = x[:, release] >= 0
    cleared = x[:, clear] >= 0
    print(
        f"STACK_RELEASE_CLEAR release_success={rate(released):.3f} "
        f"release_given_place={conditional_rate(released, placed):.3f} "
        f"body_clear_success={rate(cleared):.3f} "
        f"clear_given_release={conditional_rate(cleared, released):.3f}"
    )
    if base_disp is not None:
        disp = x[:, base_disp]
        print(
            f"STACK_BASE_DISPLACEMENT_XY_MAX mean={disp.mean():.4f} "
            f"median={np.median(disp):.4f} p95={np.quantile(disp, 0.95):.4f}"
        )
    else:
        print("STACK_BASE_DISPLACEMENT_XY_MAX unavailable=legacy_49_columns")

if len(sys.argv) >= 3:
    baseline_path = sys.argv[2]
    baseline = np.load(baseline_path)
    if baseline.ndim != 2 or baseline.shape[1] < 49:
        raise RuntimeError(
            f"baseline stack stage metrics need at least 49 columns, got {baseline.shape}"
        )
    fields = []
    for name, role in (("rehearsal", -1), ("base", 1), ("top", 0)):
        current_rate = pickup_rate(m, role)
        baseline_rate = pickup_rate(baseline, role)
        fields.append(
            f"{name}_current={current_rate:.3f} "
            f"{name}_baseline={baseline_rate:.3f} "
            f"{name}_drop={baseline_rate - current_rate:+.3f}"
        )
    print(
        f"STACK_PICKUP_REGRESSION baseline={baseline_path} " + " ".join(fields)
    )
