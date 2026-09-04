#!/usr/bin/env python3
"""Collapse the repeat rows in results.tsv into one line per run, with spread.

    python scripts/steer/summarize.py [prefix]

A single eval is not a measurement of an arm, it is a measurement of one draw of
paths. The F4 lookahead sweep read as non-monotone until every arm was scored
four times; two of the three wrong calls made that day came from ranking single
numbers. This prints mean +/- sd and n so that ordering claims have to survive
their own error bars.
"""

import re
import statistics as st
import sys
from pathlib import Path

RESULTS = Path("/home/hwanhee/CVPR2027/runs/queue/results.tsv")
COLS = ["success", "app_xt", "app_R2", "car_xt", "car_R2"]


def load():
    runs = {}
    for line in RESULTS.read_text().splitlines():
        p = line.split("\t")
        if not p or p[0] == "run":
            continue
        base = re.sub(r"_r\d+$", "", p[0])
        vals = []
        for v in p[1:6]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
        runs.setdefault(base, []).append(vals)
    return runs


def main(prefix=""):
    runs = {k: v for k, v in load().items() if k.startswith(prefix)}
    order = sorted(runs, key=lambda k: st.mean([r[1] for r in runs[k] if r[1] is not None] or [9]))
    print(f"{'run':14s} {'n':>2}  {'success':>16}  {'approach xt':>16}  {'carry xt':>16}")
    for k in order:
        rows = runs[k]
        out = [f"{k:14s} {len(rows):2d}"]
        for i in (0, 1, 3):
            vs = [r[i] for r in rows if r[i] is not None]
            if not vs:
                out.append(f"{'-':>16}"); continue
            sd = st.stdev(vs) if len(vs) > 1 else 0.0
            out.append(f"{st.mean(vs):8.4f} ±{sd:6.4f}")
        print("  ".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
