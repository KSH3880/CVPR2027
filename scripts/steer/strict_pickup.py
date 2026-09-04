#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def max_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def episode_records(data, lift_threshold=0.10, consecutive_steps=10, min_steps=30):
    required = {"box", "episode", "held", "ep0", "ep1", "z00", "z01"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"missing strict-pickup fields: {', '.join(missing)}")

    box = data["box"]
    episode = data["episode"]
    held = data["held"].astype(bool)
    records = []

    for env_id in range(episode.shape[1]):
        seen = set()
        for slot in (0, 1):
            episode_id = int(data[f"ep{slot}"][env_id])
            if episode_id < 0 or episode_id in seen:
                continue
            seen.add(episode_id)

            indices = np.flatnonzero(episode[:, env_id] == episode_id)
            if len(indices) < min_steps or np.any(np.diff(indices) != 1):
                continue

            close = held[indices, env_id]
            lift = box[indices, env_id, 2] - float(data[f"z0{slot}"][env_id])
            lifted_close = close & (lift >= lift_threshold)
            max_streak = max_true_run(lifted_close)
            complete = indices[0] > 0 and indices[-1] < episode.shape[0] - 1
            records.append({
                "env": env_id,
                "episode": episode_id,
                "steps": int(len(indices)),
                "complete": bool(complete),
                "pickup": bool(max_streak >= consecutive_steps),
                "close_any": bool(close.any()),
                "lifted_close_steps": int(lifted_close.sum()),
                "max_streak": int(max_streak),
                "max_lift": float(lift.max()),
            })
    return records


def summarize_npz(path, hand_threshold=0.35, lift_threshold=0.10,
                  consecutive_steps=10, min_steps=30):
    with np.load(path) as data:
        records = episode_records(data, lift_threshold, consecutive_steps, min_steps)

    complete = [record for record in records if record["complete"]]
    if not complete:
        raise ValueError("no complete logged episodes for strict pickup")

    def mean(key, rows):
        return float(np.mean([record[key] for record in rows]))

    return {
        "run": Path(path).stem,
        "definition": {
            "mean_hand_box_distance_lt_m": hand_threshold,
            "box_lift_from_reset_ge_m": lift_threshold,
            "consecutive_policy_steps": consecutive_steps,
        },
        "episodes": len(complete),
        "pickup_frac": mean("pickup", complete),
        "close_frac": mean("close_any", complete),
        "close_only_frac": mean("close_any", complete) - mean("pickup", complete),
        "max_lift_median": float(np.median([record["max_lift"] for record in complete])),
        "max_streak_median": float(np.median([record["max_streak"] for record in complete])),
        "observed_episodes": len(records),
        "observed_pickup_frac": mean("pickup", records) if records else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", nargs="+")
    parser.add_argument("--hand", type=float, default=0.35)
    parser.add_argument("--lift", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--min-episode-steps", type=int, default=30)
    parser.add_argument("--json")
    args = parser.parse_args()

    rows = [summarize_npz(path, args.hand, args.lift, args.steps,
                          args.min_episode_steps) for path in args.npz]
    for row in rows:
        print(
            f"STRICT_PICKUP run={row['run']} pick={row['pickup_frac']:.4f} "
            f"eps={row['episodes']} close={row['close_frac']:.4f} "
            f"closeOnly={row['close_only_frac']:.4f} "
            f"maxLift50={row['max_lift_median']:.3f} "
            f"streak50={row['max_streak_median']:.0f} "
            f"def=h{args.hand:g}_l{args.lift:g}_n{args.steps}"
        )

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows[0] if len(rows) == 1 else rows, indent=2))


if __name__ == "__main__":
    main()
