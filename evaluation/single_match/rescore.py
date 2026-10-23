"""Recompute both scores from saved position traces without Isaac Gym."""
import argparse
import json
from pathlib import Path
import numpy as np
from metrics import score_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    with np.load(args.run/'trace_0.npz', allow_pickle=False) as data:
        scores = score_trace(manifest['task'], dict(data), manifest['control_dt'])
    saved = json.loads((args.run/'results.json').read_text())['scores']
    if scores != saved:
        raise SystemExit('Saved scores differ from trace rescore')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
