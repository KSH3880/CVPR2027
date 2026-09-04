#!/usr/bin/env python3
"""Scores whole-task steer runs: approach and transport legs separately.

    python scripts/steer/analyze_full.py [runs/steer/*.npz]

The F-series path runs character -> box -> target, so a single tracking number
averages two things the policy learns very differently: walking to the box
(unloaded, the pretrained skill barely constrained) and carrying it to the
target (loaded, where every earlier result lives). Reporting one figure hides
which half moved, which is exactly what slot 7 of F1 exists to separate.

Legs are split at s_box, the arc length where the box sits; the character's
root is scored on the approach leg and the box on the transport leg.
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/hwanhee/CVPR2027")
LOGS = ROOT / "runs/queue/logs"
DS = 0.1


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


def leg_scores(pts, leg):
    """Cross-track distances and R^2 of one leg."""
    if len(pts) < 10 or len(leg) < 2:
        return None, None
    d = nearest_dist(pts, leg)
    ss_tot = ((leg - leg.mean(0)) ** 2).sum()
    return d, 1 - (d ** 2).sum() / max(ss_tot, 1e-9)


def pick_episode(d, ep, n, min_steps=30):
    """The best logged episode for env n, and the path snapshot that goes with it.

    The env keeps two path snapshots, one per episode parity. Scoring the
    newest one blindly gives nothing under --eval, where every env is reset
    together at each repeat boundary and the last episode in the window is a
    one-step stub. Take whichever slot has a real episode behind it.
    """
    best = None
    complete = False
    slots = [1, 0] if "ep1" in d else []
    for s in slots:
        eid = int(d[f"ep{s}"][n])
        if eid < 0:
            continue
        m = ep[:, n] == eid
        if m.sum() < min_steps:
            continue
        if best is None or m.sum() > best[0].sum():
            # an episode is complete only if the log window contains both its
            # start and its end -- otherwise "did the box ever reach the target"
            # is being asked of a fragment, and answers no by default
            idx = np.flatnonzero(m)
            complete = idx[0] > 0 and idx[-1] < len(m) - 1
            best = (m, d[f"gt{s}"][n], float(d[f"s_box{s}"][n]), d[f"final_tar{s}"][n], complete)
    if best is None and not slots:
        m = ep[:, n] == ep[-1, n]
        if m.sum() >= min_steps:
            s_box = float(d["s_box"][n]) if "s_box" in d else 0.0
            idx = np.flatnonzero(m)
            best = (m, d["gt"][n], s_box, d["final_tar"][n],
                    idx[0] > 0 and idx[-1] < len(m) - 1)
    return best


def score_run(npz_path):
    d = np.load(npz_path)
    root, box, ep, held = d["root"], d["box"], d["episode"], d["held"]
    legacy = bool(d["legacy"]) if "legacy" in d else True

    acc = {k: [] for k in ("app_xt", "car_xt", "app_r2", "car_r2")}
    delivered, picked, early = [], [], []
    n_approach = n_scored = 0
    for n in range(root.shape[1]):
        sel = pick_episode(d, ep, n)
        if sel is None:
            continue
        m, g_full, s_box_n, tar_n, complete = sel
        n_scored += 1
        h = held[m, n].astype(bool)
        # drop the straight-line extension the generator pads past the target
        end = int(np.linalg.norm(g_full - tar_n[:2], axis=-1).argmin()) + 1
        g = g_full[:max(end, 2)]
        i_box = int(np.clip(round(s_box_n / DS), 1, len(g) - 1))

        picked.append(h.any())
        # 38.5 % of episodes start from the carryWith reference motion, so the
        # character is already holding the box and s_box is under a metre. Those
        # have no approach to steer; averaging them in dilutes the approach
        # metrics toward zero difference no matter what the reward does.
        has_approach = s_box_n >= 0.8
        if has_approach:
            n_approach += 1
        if not legacy and has_approach and (~h).sum() >= 10:
            xt, r2 = leg_scores(root[m, n][~h], g[:i_box + 1])
            if xt is not None:
                acc["app_xt"].append(xt)
                acc["app_r2"].append(r2)
        if h.sum() >= 10:
            leg = g[i_box:] if not legacy else g
            xt, r2 = leg_scores(box[m, n][h][:, :2], leg)
            if xt is not None:
                acc["car_xt"].append(xt)
                acc["car_r2"].append(r2)

        close = np.linalg.norm(box[m, n] - tar_n, axis=-1) < 0.2
        if complete:
            delivered.append(close.any())
        if complete and close.any() and h.sum() >= 10:
            travelled = np.linalg.norm(np.diff(box[m, n][h][:, :2], axis=0), axis=-1).sum()
            leg_len = np.linalg.norm(np.diff(g[i_box:], axis=0), axis=-1).sum()
            early.append(travelled < 0.8 * max(leg_len, 1e-6))

    if not acc["car_xt"] and not acc["app_xt"]:
        return None

    def agg(key, pct=None):
        if not acc[key]:
            return None
        v = np.concatenate(acc[key]) if key.endswith("_xt") else np.asarray(acc[key])
        return float(np.percentile(v, pct)) if pct else float(np.median(v))

    return {
        "run": Path(npz_path).stem,
        "envs": int(root.shape[1]),
        "whole_task": not legacy,
        "lookahead": float(d["lookahead"]),
        "curvature": float(d["curvature"]),
        "steer": bool(d["enabled"]),
        "approach_steered": bool(d["approach"]) if "approach" in d else False,
        "app_xt_med": agg("app_xt"),
        "app_xt_p95": agg("app_xt", 95),
        "app_r2_med": agg("app_r2"),
        "car_xt_med": agg("car_xt"),
        "car_xt_p95": agg("car_xt", 95),
        "car_r2_med": agg("car_r2"),
        "approach_frac": float(n_approach / max(n_scored, 1)),
        "picked_frac": float(np.mean(picked)) if picked else 0.0,
        "delivered_frac": float(np.mean(delivered)) if delivered else None,
        "delivered_n": len(delivered),
        "early_place_frac": float(np.mean(early)) if early else 0.0,
    }


def harness_success(name):
    """Success rate from the eval log, EXCLUDING repeat 1.

    run_eval does three repeats and the first is structurally ~0 for any
    checkpointed policy, so averaging all three depresses every number by a
    third. Probe-mode runs are unaffected but are scored the same way here.
    """
    log = LOGS / f"{name}.log"
    if not log.exists():
        return None
    rates = [float(m) for m in re.findall(r"'success_rate': ([0-9.]+)", log.read_text())]
    if len(rates) >= 3:
        rates = rates[1:]
    return float(np.mean(rates)) if rates else None


def main(paths):
    rows = []
    for p in sorted(paths):
        try:
            r = score_run(p)
        except Exception as e:
            print(f"  !! {Path(p).stem}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        r["harness_success"] = harness_success(r["run"])
        rows.append(r)
        Path(p).with_suffix(".json").write_text(json.dumps(r, indent=2))

    if not rows:
        print("no scored runs")
        return

    def f(v, w=7, p=3):
        return f"{v:<{w}.{p}f}" if v is not None else " " * (w - 2) + "- "

    hdr = (f"{'run':22s} {'appr':5s} {'succ':6s} "
           f"{'app_xt':7s} {'app_R2':7s} {'car_xt':7s} {'car_R2':7s} "
           f"{'pick':6s} {'deliv':6s} {'early':6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["run"]):
        s = f"{r['harness_success']:.3f}" if r["harness_success"] is not None else "  -  "
        print(f"{r['run']:22s} {str(r['approach_steered']):5s} {s:6s} "
              f"{f(r['app_xt_med'])} {f(r['app_r2_med'])} "
              f"{f(r['car_xt_med'])} {f(r['car_r2_med'])} "
              f"{r['picked_frac']:<6.3f} {r['delivered_frac']:<6.3f} {r['early_place_frac']:<6.3f}")
    (ROOT / "runs/steer/summary_full.json").write_text(json.dumps(rows, indent=2))
    if rows:
        print(f"\napp_* is scored only on the {rows[0]['approach_frac']*100:.0f} % of episodes "
              f"with an approach leg over 0.8 m; the rest start already holding the box")
    print("\napp_* = approach leg (root vs path up to the box)")
    print("car_* = transport leg (box vs path from the box on)")


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob(str(ROOT / "runs/steer/*.npz")))
