#!/usr/bin/env python3
import sys

import numpy as np


path = sys.argv[1]
m = np.load(path)
if m.ndim != 2 or m.shape[1] < 49:
    raise RuntimeError(f"stack stage metrics need at least 49 columns, got {m.shape}")

near, pick, broken, place, release, clear, failed, max_lift, kind = range(40, 49)
base_disp = 49 if m.shape[1] >= 50 else None
diag_start = 50
diag_names = (
    "carry_steps",
    "entry_xy",
    "entry_z",
    "entry_lin",
    "entry_ang",
    "entry_upright",
    "entry_foot",
    "entry_placeable",
    "entry_joint",
    "entry_streak_max",
    "release_steps",
    "release_hand",
    "release_foot",
    "release_joint",
    "release_streak_max",
    "clear_steps",
    "clear_hand",
    "clear_body",
    "clear_stable",
    "clear_joint",
    "retreat_root_dist_max",
    "retreat_arc_max",
)


def rate(mask):
    return float(mask.mean()) if len(mask) else float("nan")


def conditional_rate(event, condition):
    return rate(event[condition])


def pickup_rate(matrix, role):
    rows = matrix[:, kind] == role
    return rate(matrix[rows, pick] >= 0)


def fraction(x, numerator, denominator):
    total = x[:, denominator].sum()
    return x[:, numerator].sum() / total if total else float("nan")


def step_count(x, numerator, denominator):
    return f"{int(x[:, numerator].sum())}/{int(x[:, denominator].sum())}"


def phase_line(name, x):
    placed = x[:, place] >= 0
    released = x[:, release] >= 0
    cleared = x[:, clear] >= 0
    clear_delay = x[cleared, clear] - x[cleared, release]
    delay_p50 = np.median(clear_delay) if len(clear_delay) else float("nan")
    return (
        f"STACK_AGENT_PHASE agent={name} base_n={len(x)} "
        f"place={placed.sum()} release={released.sum()} clear={cleared.sum()} "
        f"release_given_place={conditional_rate(released, placed):.3f} "
        f"clear_given_release={conditional_rate(cleared, released):.3f} "
        f"clear_delay_p50={delay_p50:.0f}"
    )


def gate_lines(name, x):
    d = {key: diag_start + index for index, key in enumerate(diag_names)}
    entry_streak = x[:, d["entry_streak_max"]]
    release_x = x[x[:, d["release_steps"]] > 0]
    release_streak = release_x[:, d["release_streak_max"]]
    clear_x = x[x[:, d["clear_steps"]] > 0]
    root_dist = clear_x[:, d["retreat_root_dist_max"]]
    arc = clear_x[:, d["retreat_arc_max"]]
    clear_foot = diag_start + len(diag_names)
    foot = (fraction(x, clear_foot, d["clear_steps"])
            if x.shape[1] > clear_foot else float("nan"))
    q = lambda values, p: np.quantile(values, p) if len(values) else float("nan")
    return [
        (
            f"STACK_ENTRY_GATE agent={name} "
            f"xy={fraction(x, d['entry_xy'], d['carry_steps']):.3f} "
            f"z={fraction(x, d['entry_z'], d['carry_steps']):.3f} "
            f"lin={fraction(x, d['entry_lin'], d['carry_steps']):.3f} "
            f"ang={fraction(x, d['entry_ang'], d['carry_steps']):.3f} "
            f"upright={fraction(x, d['entry_upright'], d['carry_steps']):.3f} "
            f"foot={fraction(x, d['entry_foot'], d['carry_steps']):.3f} "
            f"placeable={fraction(x, d['entry_placeable'], d['carry_steps']):.3f} "
            f"joint={fraction(x, d['entry_joint'], d['carry_steps']):.3f} "
            f"joint_steps={step_count(x, d['entry_joint'], d['carry_steps'])} "
            f"streak_p50={q(entry_streak, 0.50):.0f} "
            f"streak_p95={q(entry_streak, 0.95):.0f} max={q(entry_streak, 1.0):.0f}"
        ),
        (
            f"STACK_RELEASE_GATE agent={name} n={len(release_x)} "
            f"hand={fraction(x, d['release_hand'], d['release_steps']):.3f} "
            f"foot={fraction(x, d['release_foot'], d['release_steps']):.3f} "
            f"joint={fraction(x, d['release_joint'], d['release_steps']):.3f} "
            f"joint_steps={step_count(x, d['release_joint'], d['release_steps'])} "
            f"streak_p50={q(release_streak, 0.50):.0f} "
            f"streak_p95={q(release_streak, 0.95):.0f} max={q(release_streak, 1.0):.0f}"
        ),
        (
            f"STACK_CLEAR_GATE agent={name} n={len(clear_x)} "
            f"hand={fraction(x, d['clear_hand'], d['clear_steps']):.3f} "
            f"foot={foot:.3f} "
            f"body={fraction(x, d['clear_body'], d['clear_steps']):.3f} "
            f"stable={fraction(x, d['clear_stable'], d['clear_steps']):.3f} "
            f"joint={fraction(x, d['clear_joint'], d['clear_steps']):.3f} "
            f"joint_steps={step_count(x, d['clear_joint'], d['clear_steps'])} "
            f"root_dist_p50={q(root_dist, 0.50):.3f} "
            f"root_dist_p95={q(root_dist, 0.95):.3f} "
            f"arc_p50={q(arc, 0.50):.3f} arc_p95={q(arc, 0.95):.3f}"
        ),
    ]


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

    agent = m[:, 0].astype(int) % 2
    print(phase_line("all", x))
    for index in (0, 1):
        print(phase_line(str(index), m[base & (agent == index)]))

    if m.shape[1] >= diag_start + len(diag_names):
        for name, rows in (
            ("all", base),
            ("0", base & (agent == 0)),
            ("1", base & (agent == 1)),
        ):
            for output in gate_lines(name, m[rows]):
                print(output)
    else:
        print("STACK_GATE_DIAG unavailable=legacy_columns")

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
