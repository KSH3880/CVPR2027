import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

OUT = Path(__file__).resolve().parent
API = OUT.parents[1] / "output_etc/check/api_no_unnecessary_wait_20261111_151400/api"

SCENES = [
    "table1/mixed_2h/general_seed1/env_0000",
    "table1/mixed_4h/general_seed1/env_0036",
    "table1/mixed_8h/general_seed1/env_0000",
    "table1/mixed_16h/general_seed1/env_0000",
    "table2/coupled_2/general_seed1/env_0000",
    "table2/coupled_3/general_seed1/env_0000",
    "table2/coupled_4/general_seed1/env_0000",
    "table2/enable_sit/general_seed1/env_0000",
    "table2/enable_climb/general_seed1/env_0000",
    "table2/enable_ontop/general_seed1/env_0000",
]

ROLE_COLOR = {
    "coupled": "#2a78d6",
    "predecessor": "#eb6834",
    "follower": "#1baf7a",
    "independent": "#8a8a85",
}
INK = "#2b2b29"
MUTED = "#8f8e88"
EVENT_KEYS = ("readiness_event_id", "completion_event_id", "dependency_sensitive_event_id", "enabling_event_id")


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rect(center, dims, q):
    a = yaw(q)
    c, s = math.cos(a), math.sin(a)
    hx, hy = dims[0] / 2, dims[1] / 2
    return [(center[0] + c * dx - s * dy, center[1] + s * dx + c * dy)
            for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def agent_roles(scene):
    owner = {t[k]: t["agent_id"] for t in scene["tasks"] for k in EVENT_KEYS if k in t}
    role = {t["agent_id"]: "independent" for t in scene["tasks"]}
    for c in scene["coordination_constraints"]:
        if c["type"] == "align_events":
            for e in c["event_ids"]:
                role[owner[e]] = "coupled"
        elif c["type"] == "enabling":
            role[owner[c["enabling_event_id"]]] = "predecessor"
            role[owner[c["dependent_event_id"]]] = "follower"
    return role


def task_path(t, agent, objs):
    path = [agent["root_position_m"]]
    if t["task_label"] != "PUSH" and "object_id" in t:
        path.append(objs[t["object_id"]]["position_m"])
    if "assigned_push_region" in t:
        path.append(t["assigned_push_region"]["center_world_m"])
    elif "support_intended_final_center_m" in t:
        path.append(t["support_intended_final_center_m"])
    elif "object_goal_position_m" in t:
        path.append(t["object_goal_position_m"])
    elif "support_object_id" in t:
        path.append(objs[t["support_object_id"]]["position_m"])
    return path


def constraint_lines(scene):
    lines = []
    for c in scene["coordination_constraints"]:
        if c["type"] == "align_events":
            lines.append("align: " + ", ".join(e.split("_")[0] for e in c["event_ids"]))
        else:
            pre, post = c["enabling_event_id"].split("_"), c["dependent_event_id"].split("_")
            lines.append(f"enable: {pre[0]} {pre[1]} -> {post[0]} {post[1]}")
    return lines or ["(none)"]


def draw(ax, name, scene, small=False):
    obs = scene["initial_observation"]
    objs = {o["object_id"]: o for o in obs["objects"]}
    agents = {a["agent_id"]: a for a in obs["agents"]}
    role = agent_roles(scene)
    used = {t[k] for t in scene["tasks"] for k in ("object_id", "support_object_id") if k in t}
    fs = 6 if small else 8
    xs, ys = [], []

    for oid, o in objs.items():
        pts = rect(o["position_m"], o["dimensions_m"], o["orientation_xyzw"])
        if oid in used:
            ax.add_patch(Polygon(pts, fc="#e6e4dc", ec="#5c5b56", lw=1.0, zorder=1))
            if not small:
                ax.text(o["position_m"][0], o["position_m"][1] - 0.35, oid.replace("object_", "o"),
                        fontsize=fs - 1, color=MUTED, ha="center", va="top", zorder=2)
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
        else:
            ax.add_patch(Polygon(pts, fill=False, ec="#c4c3bc", lw=0.7, ls="--", zorder=1))

    goals = {}
    for t in scene["tasks"]:
        if "object_goal_position_m" in t:
            goals[t["object_id"]] = t["object_goal_position_m"]
    for oid, g in goals.items():
        p = objs[oid]["position_m"]
        ax.annotate("", xy=g[:2], xytext=p[:2],
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0, ls="--"), zorder=2)
        ax.plot(g[0], g[1], marker="*", ms=11 if not small else 8, color=INK, mec="white", mew=0.8, zorder=5)
        xs.append(g[0])
        ys.append(g[1])

    for t in scene["tasks"]:
        aid = t["agent_id"]
        a = agents[aid]
        col = ROLE_COLOR[role[aid]]
        path = task_path(t, a, objs)
        ax.plot([p[0] for p in path], [p[1] for p in path], ls=":", lw=1.3, color=col, zorder=3)
        if "assigned_push_region" in t:
            r = t["assigned_push_region"]
            ax.add_patch(Polygon(rect(r["center_world_m"], r["dimensions_m"], r["orientation_xyzw"]),
                                 fill=False, ec=col, lw=0.6, alpha=0.7, zorder=2))
            ax.plot(*r["center_world_m"][:2], marker="x", ms=6, mew=1.5, color=col, zorder=4)

        x, y = a["root_position_m"][:2]
        h = a["root_yaw_rad"]
        ax.annotate("", xy=(x + 0.9 * math.cos(h), y + 0.9 * math.sin(h)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4), zorder=5)
        ax.plot(x, y, "o", ms=8 if not small else 6, color=col, mec="white", mew=1.2, zorder=6)
        ax.text(x + 0.25, y + 0.25, aid if small else f"{aid} {t['task_label']}", fontsize=fs,
                color=INK, zorder=7)
        xs += [p[0] for p in path]
        ys += [p[1] for p in path]

    pad = 1.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(color="#ecebe6", lw=0.6)
    ax.tick_params(labelsize=fs - 1, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d6d5ce")
    setting = name.split("/")[1]
    title = f"{setting} | {scene['scene_id']} | {len(scene['tasks'])} agents"
    ax.set_title(title, fontsize=fs + 1, color=INK, loc="left")


def legend_handles():
    h = [Line2D([], [], marker="o", ls="", ms=8, color=c, mec="white", label=r)
         for r, c in ROLE_COLOR.items()]
    h += [
        Line2D([], [], color=INK, lw=0, marker=r"$\rightarrow$", ms=10, label="heading (root yaw)"),
        Line2D([], [], color=MUTED, ls=":", lw=1.3, label="agent -> task target"),
        Line2D([], [], marker="x", ls="", ms=6, mew=1.5, color=ROLE_COLOR["coupled"], label="assigned PUSH region"),
        Line2D([], [], marker="*", ls="", ms=11, color=INK, label="object goal"),
        Patch(fc="#e6e4dc", ec="#5c5b56", label="task object"),
        Patch(fill=False, ec="#c4c3bc", ls="--", label="distractor (no collision)"),
    ]
    return h


def main():
    loaded = [(n, json.load(open(API / n / "request.json"))["current_scene"]) for n in SCENES]

    for i, (name, scene) in enumerate(loaded, 1):
        fig, ax = plt.subplots(figsize=(8, 8))
        draw(ax, name, scene)
        ax.set_xlabel("world x (m)", fontsize=8, color=MUTED)
        ax.set_ylabel("world y (m)", fontsize=8, color=MUTED)
        ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=8, frameon=False)
        ax.text(1.04, 0.45, "constraints\n" + "\n".join(constraint_lines(scene)), transform=ax.transAxes,
                fontsize=8, color=INK, va="top", family="monospace")
        fig.savefig(OUT / f"{i:02d}_{name.split('/')[1]}_{scene['scene_id']}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(26, 11))
    for ax, (name, scene) in zip(axes.flat, loaded):
        draw(ax, name, scene, small=True)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=10, fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.08, wspace=0.25, hspace=0.25)
    fig.savefig(OUT / "overview.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
