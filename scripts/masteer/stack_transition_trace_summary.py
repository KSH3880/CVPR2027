#!/usr/bin/env python3
import sys

import numpy as np


path = sys.argv[1]
trace = np.load(path)
names = trace["names"].tolist()
data = trace["data"]
col = {name: index for index, name in enumerate(names)}

if data.ndim != 2 or data.shape[1] != len(names):
    raise RuntimeError(f"invalid stack trace shape: {data.shape}, names={len(names)}")

event = data[:, col["event"]].astype(int)
marker = data[:, col["marker"]].astype(int)
base = data[:, col["is_base"]] == 1
roles = (("base", base), ("top", ~base))


def median(mask, name):
    values = data[mask, col[name]]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


print(
    f"STACK_TRACE_SUMMARY file={path} rows={len(data)} "
    f"events={len(np.unique(event))}"
)
for role, role_mask in roles:
    before = role_mask & (marker == -1)
    after = role_mask & (marker == 0)
    fields = (
        "steer_l2", "carry0_l2", "carry1_l2", "cmd_speed",
        "action_l2", "root_speed", "root_ang_speed", "root_upright",
        "box_target_dist", "box_speed", "hand_max",
    )
    changes = " ".join(
        f"{name}={median(before, name):.3f}->{median(after, name):.3f}"
        for name in fields
    )
    print(f"STACK_TRACE_TRANSITION role={role} {changes}")

for age in (1, 5, 10, 20, 30, 40, 60):
    at_age = data[:, col["stack_age"]].astype(int) == age
    if not at_age.any():
        continue
    reached = len(np.unique(event[at_age]))
    values = []
    for role, role_mask in roles:
        mask = at_age & role_mask
        values.append(
            f"{role}[act={median(mask, 'action_l2'):.3f} "
            f"z={median(mask, 'root_z'):.3f} "
            f"v={median(mask, 'root_speed'):.3f} "
            f"upr={median(mask, 'root_upright'):.3f} "
            f"fall={median(mask, 'fallen'):.0f} "
            f"dist={median(mask, 'box_target_dist'):.3f}]"
        )
    print(f"STACK_TRACE_AGE age={age} events={reached} " + " ".join(values))

first_fall = {"base": 0, "top": 0, "both": 0, "none": 0}
max_ages = []
for event_id in np.unique(event):
    rows = event == event_id
    max_ages.append(data[rows, col["stack_age"]].max())
    fallen = rows & (data[:, col["fallen"]] == 1)
    if not fallen.any():
        first_fall["none"] += 1
        continue
    first_age = data[fallen, col["stack_age"]].min()
    first = fallen & (data[:, col["stack_age"]] == first_age)
    first_base = bool((first & base).any())
    first_top = bool((first & ~base).any())
    key = "both" if first_base and first_top else "base" if first_base else "top"
    first_fall[key] += 1

print(
    "STACK_TRACE_END "
    + " ".join(f"first_{key}={value}" for key, value in first_fall.items())
    + f" max_age_p50={np.median(max_ages):.0f}"
)
