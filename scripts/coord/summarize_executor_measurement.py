#!/usr/bin/env python3
"""Recompute an ms18 measurement summary from its saved raw NPZ."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "TokenHSI-coord"))

from coordinator.executor_calibration import summarize_measurement  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    previous = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "raw.npz") as raw:
        arrays = {key: raw[key] for key in raw.files}
    summary = summarize_measurement(arrays, previous["metadata"])
    temporary = directory / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, directory / "summary.json")
    print(json.dumps({
        "scenario": summary["metadata"]["scenario"],
        "pickup": summary["pickup_dwell_s"],
        "response": summary["response"],
        "step_response": summary["step_response"],
        "crossing": summary["crossing"],
        "collision": summary["collision"],
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
