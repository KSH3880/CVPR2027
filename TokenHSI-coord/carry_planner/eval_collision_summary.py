"""Summarize scored A=2 evaluation episodes, excluding warmup repeat zero."""

import argparse
import json
from pathlib import Path


def summarize(payload):
    groups = {}
    for record in payload["records"]:
        key = (record["repeat"], record["env"])
        agent = record["agent"]
        pair = groups.setdefault(key, {})
        if agent in pair:
            raise ValueError("duplicate scored agent episode")
        pair[agent] = record
    per_repeat = {}
    for (repeat, _), pair in groups.items():
        if set(pair) != {0, 1}:
            raise ValueError("incomplete two-agent episode")
        a, b = pair[0], pair[1]
        if a["executed_steps"] != b["executed_steps"] or a["collision_steps"] != b["collision_steps"]:
            raise ValueError("paired collision/step counts disagree")
        item = per_repeat.setdefault(repeat, dict(completed_episodes=0, collided_episodes=0,
            collision_steps=0, executed_steps=0, both_success=0))
        item["completed_episodes"] += 1
        item["collided_episodes"] += int(a["collision_steps"] > 0)
        item["collision_steps"] += a["collision_steps"]
        item["executed_steps"] += a["executed_steps"]
        item["both_success"] += int(a["success"] and b["success"])
    if set(per_repeat) != {0, 1, 2}:
        raise ValueError("expected completed evaluation repeats 0, 1 and 2")
    if len({item["completed_episodes"] for item in per_repeat.values()}) != 1:
        raise ValueError("repeat episode counts disagree")
    totals = {key: sum(per_repeat[r][key] for r in (1, 2)) for key in per_repeat[1]}
    totals.update(collision_episode_pair=totals["collided_episodes"] / totals["completed_episodes"],
                  collision_step_fraction=totals["collision_steps"] / max(totals["executed_steps"], 1),
                  both_success_fraction=totals["both_success"] / totals["completed_episodes"])
    return dict(definition=payload["definition"], threshold_m=payload["threshold_m"],
                included_repeats=[1, 2], excluded_repeats=[0], per_repeat=per_repeat, **totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.episodes.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("CARRY_COLLISION_EVAL " + json.dumps(result))


if __name__ == "__main__":
    main()
