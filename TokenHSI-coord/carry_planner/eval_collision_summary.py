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
    if "proximity_threshold_m" in payload:
        names = ("agent_agent", "agent_box", "box_box", "total")
        scored = [pair for (repeat, _), pair in groups.items() if repeat in (1, 2)]
        proximity = {}
        for name in names:
            for pair in groups.values():
                if pair[0]["proximity_steps"][name] != pair[1]["proximity_steps"][name]:
                    raise ValueError("paired proximity counts disagree")
                count = pair[0]["proximity_steps"][name]
                if not 0 <= count <= pair[0]["executed_steps"]:
                    raise ValueError("invalid proximity step count")
            counts = [pair[0]["proximity_steps"][name] for pair in scored]
            proximity[name] = dict(
                collided_episodes=sum(count > 0 for count in counts),
                collision_episode_fraction=sum(count > 0 for count in counts)/len(scored),
                collision_steps=sum(counts),
                collision_step_fraction=sum(counts)/max(totals["executed_steps"], 1))
        totals["proximity"] = proximity
        totals["proximity_threshold_m"] = payload["proximity_threshold_m"]
        totals["proximity_definition"] = payload["proximity_definition"]
    if "physical_contact_definition" in payload:
        physical = {}
        scored = [pair for (repeat, _), pair in groups.items() if repeat in (1, 2)]
        for name in ("agent_agent", "agent_box", "box_box", "total"):
            for pair in groups.values():
                count = pair[0]["physical_contact_steps"][name]
                if count != pair[1]["physical_contact_steps"][name] or not 0 <= count <= pair[0]["executed_steps"]:
                    raise ValueError("invalid paired physical contact counts")
            counts = [pair[0]["physical_contact_steps"][name] for pair in scored]
            physical[name] = dict(contacted_episodes=sum(x > 0 for x in counts),
                contact_episode_fraction=sum(x > 0 for x in counts)/len(scored),
                contact_steps=sum(counts), contact_step_fraction=sum(counts)/max(totals["executed_steps"], 1))
        totals["physical_contact"] = physical
        totals["physical_contact_definition"] = payload["physical_contact_definition"]
        totals["physical_contact_force_threshold"] = payload["physical_contact_force_threshold"]
        if all("physical_raw_contacts" in pair[0] for pair in scored):
            raw = sum(pair[0]["physical_raw_contacts"] for pair in scored)
            positive = sum(pair[0]["physical_positive_contacts"] for pair in scored)
            if raw == 0 or positive == 0:
                raise ValueError("no positive raw contacts observed; physical contact measurement unvalidated")
            totals["physical_contact_raw_records"] = raw
            totals["physical_contact_positive_records"] = positive
    scored_rows = [record for record in payload["records"] if record["repeat"] in (1, 2)]
    if scored_rows and all("speed_moments" in r for r in scored_rows):
        try:
            from carry_planner.speed_metrics import summarize_speed
        except ModuleNotFoundError:
            from speed_metrics import summarize_speed
        totals["speed_statistics"] = summarize_speed([r["speed_moments"] for r in scored_rows])
    return dict(baseline=payload.get("baseline", "none"), definition=payload["definition"], threshold_m=payload["threshold_m"],
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
