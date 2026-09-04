#!/usr/bin/env python3
"""Turns one finished evaluation into a single results row.

    python scripts/steer/record_result.py f3_base

Both numbers go in the row on purpose. Success rate alone gave the wrong answer
once already: in F2 the cross-track term looked unnecessary by success rate
(0.002 apart, under the 0.007 seed noise) while it cut approach deviation from
0.214 m to 0.127 m. A row without the tracking column would have retired it.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/hwanhee/CVPR2027")
LOGS = ROOT / "runs/queue/logs"
sys.path.insert(0, str(ROOT / "scripts"))

from queue_results import append


def success(tag):
    """Harness success rate, EXCLUDING repeat 1.

    run_eval does three repeats and the first is structurally ~0 for any
    checkpointed policy, so averaging all three understates every run by a third.
    """
    log = LOGS / f"eval_{tag}.log"
    if not log.exists():
        return None
    r = [float(m) for m in re.findall(r"'success_rate': ([0-9.]+)", log.read_text(errors="ignore"))]
    if len(r) != 3:
        # Anything else means the log is short or two evals wrote into it. Both
        # produce a number that looks fine and is wrong, so refuse instead: a
        # missing row is obvious, a quietly averaged partial run is not.
        raise SystemExit(f"record_result: {tag} has {len(r)} success_rate lines, expected 3")
    return sum(r[1:]) / 2


def settings(tag):
    """The knobs this run actually trained under, read from its own process log."""
    log = LOGS / f"{tag}.log"
    if not log.exists():
        return {}
    txt = log.read_text(errors="ignore")[:4000]
    return dict(re.findall(r"(STEER_[A-Z_]+)=([^\s]+)", txt))


def main(tag):
    npz = ROOT / f"runs/steer/{tag}.npz"
    row = {"run": tag, "success": success(tag)}
    if npz.exists():
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts/steer/analyze_full.py"), str(npz)],
            capture_output=True, text=True).stdout
        js = npz.with_suffix(".json")
        if js.exists():
            d = json.loads(js.read_text())
            row.update({k: d.get(k) for k in
                        ("app_xt_med", "app_r2_med", "car_xt_med", "car_r2_med",
                         "picked_frac", "delivered_frac", "approach_frac")})
            # delivered_frac only counts episodes whose start AND end are both
            # inside the log window, and the path snapshot has two parity slots,
            # so the sample is thin and varies run to run: f4_base landed on a
            # single episode and printed 0.0000, which reads as total failure
            # next to a harness success rate of 0.897. Blank it rather than let
            # a one-episode ratio sit in a column of real numbers.
            if (d.get("delivered_n") or 0) < 10:
                row["delivered_frac"] = None
    def f(v):
        return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    fields = [row["run"], f(row.get("success")), f(row.get("app_xt_med")),
              f(row.get("app_r2_med")), f(row.get("car_xt_med")),
              f(row.get("car_r2_med")), f(row.get("picked_frac")),
              f(row.get("delivered_frac"))]
    append(fields)


if __name__ == "__main__":
    main(sys.argv[1])
