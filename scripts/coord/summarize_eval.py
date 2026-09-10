#!/usr/bin/env python3
"""Summarize one deterministic C1 closed-loop evaluation.

The raw ``MA_METRICS`` rows are the source of truth.  This script keeps the
compact JSON/one-line contract used by the overnight experiment loop.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _mean(value):
    return float(np.mean(value)) if len(value) else float("nan")


def _median(value):
    return float(np.median(value)) if len(value) else float("nan")


def _paired(values, rows, predicate="both"):
    """Pair adjacent a0/a1 rows without collapsing the three eval repeats."""
    if len(values) % 2:
        return float("nan")
    order_ok = np.all((rows[0::2] % 2 == 0) & (rows[1::2] % 2 == 1))
    env_ok = np.all(rows[0::2] // 2 == rows[1::2] // 2)
    if not order_ok or not env_ok:
        return float("nan")
    pair = values.reshape(-1, 2)
    if predicate == "both":
        result = pair[:, 0] & pair[:, 1]
    elif predicate == "one":
        result = pair[:, 0] ^ pair[:, 1]
    elif predicate == "any":
        result = pair[:, 0] | pair[:, 1]
    else:
        raise ValueError(predicate)
    return float(result.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    matrix = np.load(args.metrics)
    if matrix.ndim != 2 or matrix.shape[1] < 50:
        raise ValueError(f"expected MA metrics [N,>=50], got {matrix.shape}")
    text = args.log.read_text(errors="ignore")
    rates = [float(value) for value in re.findall(r"'success_rate': ([0-9.]+)", text)]
    stable_rates = rates[-3:] if len(rates) >= 3 else rates

    rows = matrix[:, 0].astype(int)
    finished = matrix[:, 1] >= 0
    collided = matrix[:, 2] > 0
    steps = np.maximum(matrix[:, 30], 1.0)
    moving_steps = np.maximum(matrix[:, 30] - matrix[:, 4], 1.0)
    grasped = matrix[:, 44] >= 0
    delivered = matrix[:, 45] > 0
    putdown = matrix[:, 46] > 0
    carried = matrix[:, 47] > 0
    placed = matrix[:, 48] > 0
    planned_bin_steps = matrix[:, 35:39].sum(axis=0)
    has_sent_metrics = matrix.shape[1] >= 58
    command_bin_steps = (
        matrix[:, 52:56].sum(axis=0) if has_sent_metrics else planned_bin_steps
    )
    command_steps = max(float(command_bin_steps.sum()), 1.0)
    lower_command_steps = (
        matrix[:, 52:55].sum(axis=1) if has_sent_metrics
        else matrix[:, 35:38].sum(axis=1)
    )
    used_lower_command = lower_command_steps > 0

    summary = {
        "tag": args.tag,
        "scenario": args.scenario,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes_agent": int(len(matrix)),
        "success_rate": _mean(stable_rates),
        "finished": _mean(finished),
        "finish_step_median": _median(matrix[finished, 1]),
        "collision_episode_agent": _mean(collided),
        "collision_episode_pair": _paired(collided, rows, "any"),
        "collision_steps_mean": _mean(matrix[:, 2]),
        "root_min_distance_median": _median(matrix[:, 6]),
        "path_median": _median(matrix[:, 5]),
        "lat_root_median": _median(matrix[:, 26] / steps),
        "lat_box_median": _median(matrix[:, 27] / steps),
        "speed_error_median": _median(matrix[:, 28] / steps),
        "gait_speed_median": _median(matrix[:, 5] / (moving_steps / 30.0)),
        "command_speed_bin_steps": [float(value) for value in command_bin_steps],
        "planned_speed_bin_steps": [float(value) for value in planned_bin_steps],
        "speed_bins_are_rate_limited_command": bool(has_sent_metrics),
        "lower_command_fraction": float(command_bin_steps[:3].sum() / command_steps),
        "lower_command_episode_agent": _mean(used_lower_command),
        "lower_command_episode_pair_any": _paired(used_lower_command, rows, "any"),
        "lower_command_episode_pair_one": _paired(used_lower_command, rows, "one"),
        "grasp_episode": _mean(grasped),
        "delivered": _mean(delivered),
        "putdown": _mean(putdown),
        "carry": _mean(carried),
        "place": _mean(placed),
        "both_carry": _paired(carried, rows),
        "both_place": _paired(placed, rows),
        "one_carry": _paired(carried, rows, "one"),
        "shortcut": _mean(matrix[:, 49] > 0),
    }
    if has_sent_metrics:
        summary.update({
            "collision_steps_approach_mean": _mean(matrix[:, 50]),
            "collision_steps_carry_mean": _mean(matrix[:, 51]),
            "sent_lower_steps_approach": float(matrix[:, 56].sum()),
            "sent_lower_steps_carry": float(matrix[:, 57].sum()),
        })
    if matrix.shape[1] >= 64:
        # Pair counters are duplicated on adjacent agent rows; count one copy
        # per environment episode.  Modes are [neither, exactly one, both].
        pair_mode_steps = matrix[0::2, 58:61].sum(axis=0)
        pair_collision_mode_steps = matrix[0::2, 61:64].sum(axis=0)
        pair_steps = max(float(pair_mode_steps.sum()), 1.0)
        mode_fraction = pair_mode_steps / pair_steps
        collision_rate = np.divide(
            pair_collision_mode_steps,
            pair_mode_steps,
            out=np.full(3, np.nan),
            where=pair_mode_steps > 0,
        )
        summary.update({
            "command_pair_mode_steps": [float(value) for value in pair_mode_steps],
            "command_pair_mode_fraction": [float(value) for value in mode_fraction],
            "collision_pair_mode_steps": [
                float(value) for value in pair_collision_mode_steps
            ],
            "collision_rate_by_command_mode": [
                float(value) for value in collision_rate
            ],
            "one_command_step_fraction": float(mode_fraction[1]),
            "both_command_step_fraction": float(mode_fraction[2]),
            "collision_rate_one_command": float(collision_rate[1]),
            "collision_rate_both_command": float(collision_rate[2]),
        })
    runtime = re.findall(
        r"COORD_SUMMARY provider=\S+ replans=(\d+) invalid=(\d+) "
        r"invalid_rate=([0-9.eE+-]+) unsafe_selected=(\d+) analytic_fallback=(\d+)",
        text,
    )
    if runtime:
        replans, invalid, invalid_rate, unsafe, fallback = runtime[-1]
        summary.update({
            "replans": int(replans),
            "invalid": int(invalid),
            "invalid_rate": float(invalid_rate),
            "unsafe_selected": int(unsafe),
            "unsafe_rate": int(unsafe) / max(int(replans), 1),
            "analytic_fallback": int(fallback),
        })
    invalid_reasons = re.findall(
        r"COORD_INVALID_REASONS finite=(\d+) anchors=(\d+) "
        r"buffer=(\d+) speed=(\d+) curve=(\d+)",
        text,
    )
    if invalid_reasons:
        finite, anchors, buffer, speed, curve = (
            int(value) for value in invalid_reasons[-1]
        )
        reason_counts = {
            "finite": finite,
            "anchors": anchors,
            "buffer": buffer,
            "speed": speed,
            "curve": curve,
        }
        denominator = max(int(summary.get("replans", 0)), 1)
        summary.update({
            "invalid_reason_counts": reason_counts,
            "invalid_finite_rate": finite / denominator,
            "invalid_anchor_rate": anchors / denominator,
            "invalid_buffer_rate": buffer / denominator,
            "invalid_speed_rate": speed / denominator,
            "invalid_curve_rate": curve / denominator,
        })
    curve_by_held = re.findall(
        r"COORD_CURVE_INVALID_BY_HELD held0=(\d+) held1=(\d+) held2=(\d+)",
        text,
    )
    if curve_by_held:
        held0, held1, held2 = (int(value) for value in curve_by_held[-1])
        summary["invalid_curve_by_held_count"] = [held0, held1, held2]
    candidate = re.findall(
        r"COORD_CANDIDATES choices=([0-9,]+) diversity=([0-9.eE+-]+) "
        r"min_hh=([0-9.eE+-]+) bb_margin=([0-9.eE+-]+) "
        r"hb_margin=([0-9.eE+-]+) makespan=([0-9.eE+-]+) "
        r"pred_collision=([0-9.eE+-]+)",
        text,
    )
    if candidate:
        choices, diversity, min_hh, bb_margin, hb_margin, makespan, collision = candidate[-1]
        summary.update({
            "candidate_choices": [int(value) for value in choices.split(",")],
            "candidate_diversity": float(diversity),
            "predicted_min_hh": float(min_hh),
            "predicted_bb_margin": float(bb_margin),
            "predicted_hb_margin": float(hb_margin),
            "predicted_makespan": float(makespan),
            "predicted_collision": float(collision),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = (
        "success_rate", "carry", "place", "both_carry", "both_place",
        "collision_episode_pair", "root_min_distance_median", "lat_root_median",
        "speed_error_median", "invalid_rate", "unsafe_rate",
        "candidate_diversity", "lower_command_fraction",
        "lower_command_episode_pair_one",
        "one_command_step_fraction", "both_command_step_fraction",
        "collision_rate_one_command", "collision_rate_both_command",
        "invalid_curve_rate",
    )
    compact = " ".join(
        f"{key}={summary[key]:.4f}" for key in fields if key in summary
    )
    print(f"COORD_EVAL_SUMMARY tag={args.tag} scen={args.scenario} {compact}")


if __name__ == "__main__":
    main()
