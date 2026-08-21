#!/usr/bin/env python3
"""Scores steer runs offline: executed path vs the ground-truth path.

    python scripts/steer/analyze.py [runs/steer/*.npz]

Emits one row per run plus a json summary next to the npz.
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/hwanhee/CVPR2027")
LOGS = ROOT / "runs/queue/logs"


def nearest_dist(pts, path):
    """Min distance from each point [T,2] to a polyline [V,2]."""
    a, b = path[:-1], path[1:]
    ab = b - a
    denom = (ab * ab).sum(-1)
    denom[denom < 1e-12] = 1e-12
    t = ((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / denom[None]
    t = np.clip(t, 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(axis=1)


def score_run(npz_path):
    d = np.load(npz_path)
    root, box, gt = d["root"], d["box"], d["gt"]
    ep, prog, held = d["episode"], d["progress"], d["held"]
    tar = d["final_tar"]
    T, N = root.shape[0], root.shape[1]

    xt_all, r2_all, early, delivered = [], [], [], []
    for n in range(N):
        # gt and final_tar are snapshots of the LAST episode, so only that
        # episode can be scored against them
        m = ep[:, n] == ep[-1, n]
        if m.sum() < 30:
            continue
        # the gt is the TRANSPORT path (box -> target); score the box against it
        # over the steps where steering was actually governing
        act = d["active"][m, n].astype(bool) if "active" in d else held[m, n].astype(bool)
        if act.sum() < 15:
            continue
        r = box[m, n][act][:, :2]
        g = gt[n]
        # trim the gt tail that is pure straight-line extension past the target
        keep = np.linalg.norm(g - tar[n, :2], axis=-1).argmin() + 1
        g = g[:max(keep, 2)]

        dist = nearest_dist(r, g)
        xt_all.append(dist)
        ss_res = (dist ** 2).sum()
        ss_tot = ((g - g.mean(0)) ** 2).sum()
        r2_all.append(1 - ss_res / max(ss_tot, 1e-9))

        b = box[m, n]
        close = np.linalg.norm(b - tar[n], axis=-1) < 0.2
        delivered.append(close.any())
        if close.any():
            first = close.argmax()
            travelled = np.linalg.norm(np.diff(r[:first + 1], axis=0), axis=-1).sum()
            gt_len = np.linalg.norm(np.diff(g, axis=0), axis=-1).sum()
            early.append(travelled < 0.8 * gt_len)

    if not xt_all:
        return None
    xt = np.concatenate(xt_all)
    return {
        "run": Path(npz_path).stem,
        "envs": N,
        "lookahead": float(d["lookahead"]),
        "handoff": float(d["handoff"]),
        "curvature": float(d["curvature"]),
        "steer": bool(d["enabled"]),
        "path_r2_med": float(np.median(r2_all)),
        "path_r2_p10": float(np.percentile(r2_all, 10)),
        "xtrack_mean": float(xt.mean()),
        "xtrack_p95": float(np.percentile(xt, 95)),
        "delivered_frac": float(np.mean(delivered)),
        "early_place_frac": float(np.mean(early)) if early else 0.0,
        "held_frac": float(held.mean()),
    }


def harness_success(name):
    """The success_rate the eval harness printed, averaged over object sets."""
    log = LOGS / f"{name}.log"
    if not log.exists():
        return None
    rates = [float(m) for m in re.findall(r"'success_rate': ([0-9.]+)", log.read_text())]
    return float(np.mean(rates)) if rates else None


def main(paths):
    rows = []
    for p in sorted(paths):
        try:
            r = score_run(p)
        except Exception as e:
            print(f"  !! {Path(p).stem}: {e}")
            continue
        if r is None:
            continue
        r["harness_success"] = harness_success(r["run"])
        rows.append(r)
        Path(p).with_suffix(".json").write_text(json.dumps(r, indent=2))

    if not rows:
        print("no scored runs")
        return
    hdr = f"{'run':24s} {'steer':5s} {'la':4s} {'curv':5s} {'succ':6s} {'R2med':7s} {'xt_mean':8s} {'xt_p95':7s} {'deliv':6s} {'early':6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["run"]):
        s = f"{r['harness_success']:.3f}" if r["harness_success"] is not None else "  -  "
        print(f"{r['run']:24s} {str(r['steer']):5s} {r['lookahead']:<4.1f} {r['curvature']:<5.2f} "
              f"{s:6s} {r['path_r2_med']:<7.3f} {r['xtrack_mean']:<8.3f} {r['xtrack_p95']:<7.3f} "
              f"{r['delivered_frac']:<6.3f} {r['early_place_frac']:<6.3f}")
    (ROOT / "runs/steer/summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:] or glob.glob(str(ROOT / "runs/steer/*.npz"))
    main(args)
