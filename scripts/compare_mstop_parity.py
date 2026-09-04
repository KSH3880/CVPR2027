#!/usr/bin/env python3

import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def array(record, key):
    value = record.get(key)
    return None if value is None else np.asarray(value, dtype=np.float64)


def max_abs(left, right):
    if left is None or right is None:
        return None
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right)))


def base_action(record):
    value = array(record, "base_action")
    return value if value is not None else array(record, "mu")


def final_action(record):
    value = array(record, "final_action")
    return value if value is not None else array(record, "mu")


def first_over(rows, key, tol):
    for i, row in enumerate(rows):
        value = row[key]
        if value is not None and value > tol:
            return i, value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a_trace")
    parser.add_argument("b_trace")
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    left = load(args.a_trace)
    right = load(args.b_trace)
    count = min(len(left), len(right))
    if count == 0:
        raise SystemExit("비교할 parity record가 없습니다")

    rows = []
    for i in range(count):
        a, b = left[i], right[i]
        self_size = min(a["self_obs_size"], b["self_obs_size"])
        a_raw, b_raw = array(a, "raw_obs"), array(b, "raw_obs")
        rows.append({
            "raw": max_abs(a_raw, b_raw),
            "normalized": max_abs(array(a, "normalized_obs"), array(b, "normalized_obs")),
            "self": max_abs(a_raw[..., :self_size], b_raw[..., :self_size]),
            "task_head": max_abs(array(a, "task_head"), array(b, "task_head")),
            "base": max_abs(base_action(a), base_action(b)),
            "b_gate": max_abs(array(b, "gate"), np.zeros_like(array(b, "gate"))) if array(b, "gate") is not None else None,
            "b_delta": float(np.max(np.abs(array(b, "delta")))) if array(b, "delta") is not None else None,
            "b_final_base": max_abs(final_action(b), base_action(b)),
        })

    print(f"records A/B: {len(left)}/{len(right)}, compared: {count}, tol: {args.tol:g}")
    if rows[0]["self"] is not None and rows[0]["self"] > args.tol:
        print("WARNING: step-0 self observation이 다릅니다; action보다 reset/seed parity를 먼저 확인하세요")

    print("step 0:")
    for key, value in rows[0].items():
        print(f"  {key:14s} {value}")

    print("first value above tolerance:")
    for key in ("raw", "normalized", "self", "task_head", "base", "b_gate", "b_final_base"):
        found = first_over(rows, key, args.tol)
        print(f"  {key:14s} {found}")

    print("first 10 actor calls:")
    print(" step       raw      self  task_head      base    b_gate  b_final-base")
    for i, row in enumerate(rows[:10]):
        values = [row[k] for k in ("raw", "self", "task_head", "base", "b_gate", "b_final_base")]
        shown = ["None" if v is None else f"{v:.3e}" for v in values]
        print(f"{i:5d} " + " ".join(f"{v:>9s}" for v in shown))


if __name__ == "__main__":
    main()
