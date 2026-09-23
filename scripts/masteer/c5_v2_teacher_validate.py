#!/usr/bin/env python3
"""Analyze and visualize mode-labeled C5 teacher elites."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))

from coordinator.planner import candidate_costs, timed_rollout
from coordinator.schema import CoordinatorState
from coordinator_c5_v2 import C5TrajectoryCodec


MODE_NAMES = {1: "Left", 2: "Right", 3: "Slow"}
MODE_COLORS = {"Left": "#2563a6", "Right": "#d97706", "Slow": "#7a8b32"}


def gather(value, index):
    row = torch.arange(value.shape[0])
    return value[row, index]


def first_entry_time(position, query_time, center, radius):
    distance = (position - center[:, None, None, :]).norm(dim=-1)
    inside = distance <= radius
    has_entry = inside.any(dim=-1)
    first = inside.float().argmax(dim=-1)
    time = query_time.gather(-1, first[..., None]).squeeze(-1)
    return torch.where(has_entry, time, torch.full_like(time, float("nan")))


def analyze_record(record, pair_offset, codec, conflict_radius):
    state = CoordinatorState.from_mapping(record["state"])
    action = record["action"]
    decoded = codec(action, state)
    diagnostics = candidate_costs(decoded, state, measured_executor_timing=True)
    safe = diagnostics["safe"] & diagnostics["valid"]
    masked_score = record["score"].masked_fill(~safe, float("inf"))
    best = masked_score.argmin(dim=1)
    has_safe = safe.any(dim=1)
    path = gather(decoded["path_world"], best)
    speed = gather(decoded["speed"], best)
    dwell = gather(decoded["pickup_dwell"], best)
    selected = {key: gather(value, best) for key, value in diagnostics.items()
                if value.ndim >= 2 and value.shape[:2] == safe.shape}
    rollout = timed_rollout(
        path[:, None], speed[:, None], dwell[:, None], state,
        measured_executor_timing=True,
    )
    center = 0.5 * (state.root_xy + state.goal_xy).mean(dim=1)
    entry = first_entry_time(
        rollout["root"][:, 0], rollout["query_time"][:, 0], center,
        conflict_radius,
    )
    direct = (state.goal_xy - state.root_xy).norm(dim=-1).sum(dim=-1)
    mode = MODE_NAMES[int(record["maneuver"][0].item())]
    rows = []
    for index in range(state.batch_size):
        clearance = min(
            float(selected["min_hh"][index]) - 1.0,
            float(selected["min_bb_margin"][index]),
            float(selected["min_hb_margin"][index]),
        )
        t0, t1 = float(entry[index, 0]), float(entry[index, 1])
        if np.isnan(t0) or np.isnan(t1):
            order = "no-entry"
            gap = float("nan")
        else:
            gap = abs(t0 - t1)
            order = "A-first" if t0 < t1 else "B-first" if t1 < t0 else "tie"
        rows.append({
            "pair_id": pair_offset + index,
            "geometry_id": (pair_offset + index) // 2,
            "role": int(record["role"][index]),
            "mode": mode,
            "has_safe_elite": bool(has_safe[index]),
            "best_elite": int(best[index]),
            "joint_path_length_m": float(selected["length"][index]),
            "detour_ratio": float(selected["length"][index] / direct[index]),
            "makespan_s": float(selected["makespan"][index]),
            "min_hh_m": float(selected["min_hh"][index]),
            "clearance_margin_m": clearance,
            "entry_a_s": t0,
            "entry_b_s": t1,
            "entry_gap_s": gap,
            "entry_order": order,
            "score": float(masked_score[index, best[index]]),
            "path": path[index].cpu(),
            "speed": speed[index].cpu(),
            "center": center[index].cpu(),
        })
    return rows


def load_rows(payload, conflict_radius):
    codec = C5TrajectoryCodec()
    rows = []
    pair_offset = 0
    batches = payload["batches"]
    for start in range(0, len(batches), 3):
        group = batches[start:start + 3]
        batch_size = group[0]["role"].shape[0]
        for record in group:
            rows.extend(analyze_record(record, pair_offset, codec, conflict_radius))
        pair_offset += batch_size
    return rows


def summarize(rows):
    summary = {}
    for mode in MODE_NAMES.values():
        values = [row for row in rows if row["mode"] == mode and row["has_safe_elite"]]
        mode_summary = {"safe_pairs": len(values)}
        for field in ("joint_path_length_m", "detour_ratio", "makespan_s",
                      "clearance_margin_m", "entry_gap_s"):
            array = np.asarray([row[field] for row in values], dtype=np.float64)
            array = array[np.isfinite(array)]
            mode_summary[field] = {
                "median": float(np.median(array)),
                "p10": float(np.quantile(array, 0.1)),
                "p90": float(np.quantile(array, 0.9)),
            } if array.size else None
        mode_summary["entry_order"] = {
            name: sum(row["entry_order"] == name for row in values)
            for name in ("A-first", "B-first", "tie", "no-entry")
        }
        summary[mode] = mode_summary
    return summary


def plot_paths(rows, output, geometry_id, role, conflict_radius):
    selected = [row for row in rows if row["geometry_id"] == geometry_id and row["role"] == role]
    selected.sort(key=lambda row: list(MODE_NAMES.values()).index(row["mode"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for axis, row in zip(axes, selected):
        path = row["path"].numpy()
        center = row["center"].numpy()
        circle = plt.Circle(center, conflict_radius, fill=False, color="#666666",
                            linestyle=":", linewidth=1.2)
        axis.add_patch(circle)
        axis.plot(path[0, :, 0], path[0, :, 1], color="#2563a6", label="Agent A")
        axis.plot(path[1, :, 0], path[1, :, 1], color="#d97706", linestyle="--",
                  label="Agent B")
        axis.scatter(path[:, 0, 0], path[:, 0, 1], marker="o", color="#222222", s=24)
        axis.scatter(path[:, -1, 0], path[:, -1, 1], marker="*", color="#222222", s=70)
        axis.set_title(row["mode"])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.text(0.02, 0.02,
                  f"makespan {row['makespan_s']:.2f}s\nclearance {row['clearance_margin_m']:.2f}m\n"
                  f"detour {row['detour_ratio']:.2f}x",
                  transform=axis.transAxes, fontsize=9, va="bottom")
    axes[0].set_ylabel("world y (m)")
    for axis in axes:
        axis.set_xlabel("world x (m)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"C5 Teacher trajectories · geometry {geometry_id}, role {role}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(rows, output):
    fields = [
        ("detour_ratio", "Detour ratio (× direct)"),
        ("makespan_s", "Makespan (s)"),
        ("clearance_margin_m", "Clearance margin (m)"),
    ]
    modes = list(MODE_NAMES.values())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, (field, label) in zip(axes, fields):
        values = [
            [row[field] for row in rows if row["mode"] == mode
             and row["has_safe_elite"] and np.isfinite(row[field])]
            for mode in modes
        ]
        plot = axis.boxplot(values, labels=modes, patch_artist=True, showfliers=False)
        for box, mode in zip(plot["boxes"], modes):
            box.set_facecolor(MODE_COLORS[mode])
            box.set_alpha(0.65)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.suptitle("C5 Teacher mode efficiency and safety · 128 state-role pairs")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_representatives(rows, count):
    selected = []
    per_mode = max(1, count // 3)
    for mode in MODE_NAMES.values():
        values = [row for row in rows if row["mode"] == mode and row["has_safe_elite"]]
        values.sort(key=lambda row: row["clearance_margin_m"])
        low_clearance = values[:max(1, per_mode // 2)]
        efficient = sorted(values, key=lambda row: row["makespan_s"])[:per_mode - len(low_clearance)]
        selected.extend(low_clearance + efficient)
    return selected[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--geometry", type=int, default=0)
    parser.add_argument("--role", type=int, choices=(0, 1), default=0)
    parser.add_argument("--conflict-radius", type=float, default=1.0)
    parser.add_argument("--representatives", type=int, default=15)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu")
    rows = load_rows(payload, args.conflict_radius)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    scalar_fields = [key for key in rows[0] if key not in ("path", "speed", "center")]
    with (args.output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)
    representatives = select_representatives(rows, args.representatives)
    with (args.output_dir / "representatives.json").open("w") as file:
        json.dump([{key: row[key] for key in scalar_fields} for row in representatives], file, indent=2)
    plot_paths(rows, args.output_dir / "paths_by_mode.png", args.geometry, args.role,
               args.conflict_radius)
    plot_metrics(rows, args.output_dir / "mode_metrics.png")
    print(json.dumps({"rows": len(rows), "summary": summary,
                      "representatives": len(representatives)}, sort_keys=True))


if __name__ == "__main__":
    main()
