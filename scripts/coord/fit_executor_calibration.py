#!/usr/bin/env python3
"""Fit the simple measured-v1 timing table from one or more Cross runs."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-coord"))

from coordinator.executor_calibration import (  # noqa: E402
    CONSTANT_SPEEDS,
    summarize_measurement,
)


TIME_KEYS = {
    "active", "requested_speed", "sent_speed", "root_velocity_speed",
    "displacement_speed", "held", "phase", "root_xy", "box_xyz",
    "goal_xy", "root_box_distance", "body_min_distance", "root_distance",
    "box_distance", "human_box_distance",
}


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else float("nan"),
        "p10": float(np.percentile(values, 10)) if values.size else float("nan"),
        "p90": float(np.percentile(values, 90)) if values.size else float("nan"),
    }


def _load_and_merge(directories):
    loaded = []
    metadata = []
    for directory in directories:
        with np.load(directory / "raw.npz") as raw:
            loaded.append({key: raw[key] for key in raw.files})
        metadata.append(json.loads((directory / "summary.json").read_text())["metadata"])
    steps = {item["active"].shape[0] for item in loaded}
    if len(steps) != 1:
        raise ValueError(f"all runs must have the same number of steps: {steps}")
    keys = set(loaded[0])
    if any(set(item) != keys for item in loaded):
        raise ValueError("raw measurement keys differ")
    merged = {}
    for key in keys:
        axis = 1 if key in TIME_KEYS else 0
        merged[key] = np.concatenate([item[key] for item in loaded], axis=axis)
    combined_meta = dict(metadata[0])
    combined_meta["envs"] = int(sum(int(item["envs"]) for item in metadata))
    combined_meta["seed"] = [int(item["seed"]) for item in metadata]
    combined_meta["sources"] = [str(path) for path in directories]
    return merged, combined_meta


def _phase1_remaining(arrays, dt):
    values = []
    held = arrays["held"]
    phase = arrays["phase"]
    active = arrays["active"]
    for env in range(held.shape[1]):
        for agent in range(held.shape[2]):
            first_held = np.flatnonzero(held[:, env, agent])
            if first_held.size == 0:
                continue
            end = int(first_held[0])
            near = np.flatnonzero(active[:end, env] & (phase[:end, env, agent] == 1))
            values.extend((end - near) * dt)
    return _stats(values)


def _project(point, start, end):
    delta = end - start
    length2 = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        np.sum((point - start) * delta, axis=-1) / np.maximum(length2, 1e-9),
        0.0, 1.0,
    )
    closest = start + fraction[..., None] * delta
    return np.linalg.norm(point - closest, axis=-1), fraction, np.sqrt(length2)


def _calibrated_crossing_diagnostics(arrays, approach_map, carry_map, dwell0, dwell1):
    command = np.asarray(CONSTANT_SPEEDS, dtype=np.float64)
    profile = arrays["profile_id"]
    constant = profile < command.size
    requested = command[np.minimum(profile, command.size - 1)][:, None] * np.ones((1, 2))
    approach_speed = np.interp(requested, command, approach_map)
    carry_speed = np.interp(requested, command, carry_map)
    root = arrays["root_xy"][0]
    box = arrays["box_xyz"][0, ..., :2]
    goal = arrays["goal_xy"][0]
    crossing = arrays["crossing_point"]
    held = arrays["held"][0]
    phase = arrays["phase"][0]
    first_distance, first_fraction, first_length = _project(crossing, root, box)
    second_distance, second_fraction, second_length = _project(crossing, box, goal)
    on_second = second_distance < first_distance
    pickup = np.where(held, 0.0, np.where(phase == 1, dwell1, dwell0))
    calibrated = np.where(
        on_second,
        first_length / approach_speed + pickup
        + second_fraction * second_length / carry_speed,
        first_fraction * first_length / approach_speed,
    )
    actual = arrays["actual_crossing_time"]
    old = arrays["predicted_crossing_time"]
    valid = constant[:, None] & (arrays["actual_crossing_distance"] < 0.75) & (actual >= 0.0)
    both = valid.all(axis=1)
    old_error = np.abs(actual[valid] - old[valid])
    calibrated_error = np.abs(actual[valid] - calibrated[valid])
    actual_gap = np.abs(actual[:, 1] - actual[:, 0])
    old_gap = np.abs(old[:, 1] - old[:, 0])
    calibrated_gap = np.abs(calibrated[:, 1] - calibrated[:, 0])
    return {
        "arrival_abs_error_before_s": _stats(old_error),
        "arrival_abs_error_calibrated_s": _stats(calibrated_error),
        "gap_abs_error_before_s": _stats(np.abs(actual_gap[both] - old_gap[both])),
        "gap_abs_error_calibrated_s": _stats(
            np.abs(actual_gap[both] - calibrated_gap[both])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    directories = [path.resolve() for path in args.directories]
    arrays, metadata = _load_and_merge(directories)
    summary = summarize_measurement(arrays, metadata)
    profile_names = [f"const_{int(value * 1000):04d}" for value in CONSTANT_SPEEDS]
    approach_map = [
        summary["profiles"][name]["approach"]["actual_speed"]["median"]
        for name in profile_names
    ]
    carry_map = [
        summary["profiles"][name]["carry"]["actual_speed"]["median"]
        for name in profile_names
    ]
    dwell = summary["pickup_dwell_s"]
    phase1 = _phase1_remaining(arrays, float(metadata["dt"]))
    diagnostics = _calibrated_crossing_diagnostics(
        arrays, approach_map, carry_map, dwell["median"], phase1["median"]
    )
    result = {
        "schema_version": "tokenhsi-ms18-timing-calibration-v1",
        "sources": [str(path) for path in directories],
        "command_speed_mps": list(CONSTANT_SPEEDS),
        "approach_actual_speed_mps": approach_map,
        "carry_actual_speed_mps": carry_map,
        "pickup_dwell_from_near_s": dwell,
        "phase1_remaining_dwell_s": phase1,
        "step_response": summary["step_response"],
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
