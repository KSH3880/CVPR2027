import json
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from viz_scenes import API, INK, MUTED, OUT, ROLE_COLOR, SCENES, agent_roles, rect

EVAL = OUT.parents[1] / "teacher_runtime/output_etc/check/eval_protocol_v1_legacy_20261111_165027/eval"
VIDEO = OUT / "videos"
FPS = 30
HOLD = 60
FROZEN = "#d9d8d2"


def load(name):
    run, env = name.rsplit("/", 1)
    e = int(env.split("_")[1])
    tr = np.load(EVAL / run / "trace.npz")
    audit = json.load(open(EVAL / run / "release_audit.json"))
    return dict(
        scene=json.load(open(API / name / "request.json"))["current_scene"],
        response=json.loads((API / name / "response.raw.txt").read_text()),
        dt=json.load(open(EVAL / run / "result.json"))["control_dt"],
        release={r["agent_id"]: r["release_frame"] for r in audit["release_events"][e]},
        roots=tr["roots"][:, e],
        boxes=tr["boxes"][:, e],
        sizes=tr["sizes"][e],
        fall=tr["fall_step"][e],
        ever=tr["ever"][e, :, 1],
        box_goal=tr["shared_center_ever"][e],
        simultaneous=tr["simultaneous"][e, :, 1],
    )


def arena_slots(boxes, sizes, roots):
    center = roots[0, :, :2].mean(0)
    keep, seen = [], []
    for k in range(boxes.shape[1]):
        key = np.round(np.r_[boxes[0, k, :2], sizes[k]], 3)
        if np.linalg.norm(boxes[0, k, :2] - center) < 100 and not any((key == s).all() for s in seen):
            keep.append(k)
            seen.append(key)
    return keep


def outcome_lines(scene, d, index):
    mark = {True: "✓", False: "✗"}
    lines = []
    for c in scene["coordination_constraints"]:
        if c["type"] == "align_events":
            ids = [ev.split("_")[0] for ev in c["event_ids"]]
            lines.append(f"Coupled {', '.join(ids)}: box reached goal {mark[bool(d['box_goal'][index[ids[0]]])]}")
        else:
            pre, post = c["enabling_event_id"].split("_")[0], c["dependent_event_id"].split("_")[0]
            lines.append(f"Enable {pre} -> {post}: simultaneous success {mark[bool(d['simultaneous'][index[post]])]}")
    return lines


def render(name):
    d = load(name)
    scene, roots, boxes, sizes, dt = d["scene"], d["roots"], d["boxes"], d["sizes"], d["dt"]
    T, N = roots.shape[:2]
    tasks = {t["agent_id"]: t for t in scene["tasks"]}
    ids = [a["agent_id"] for a in scene["initial_observation"]["agents"]]
    index = {aid: i for i, aid in enumerate(ids)}
    role = agent_roles(scene)
    color = [ROLE_COLOR[role[a]] for a in ids]
    release = [d["release"][a] for a in ids]
    slots = arena_slots(boxes, sizes, roots)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[2.5, 0.55, 0.85], left=0.05, right=0.97,
                          top=0.9, bottom=0.04, wspace=0.18, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    tl = fig.add_subplot(gs[0, 1])
    info = fig.add_subplot(gs[1, 1])
    key = fig.add_subplot(gs[2, 1])
    info.axis("off")
    key.axis("off")
    setting = name.split("/")[1]
    fig.suptitle(f"{setting} | {scene['scene_id']} | {N} agents", x=0.05, ha="left", fontsize=13, color=INK)
    fig.text(0.05, 0.925, "Replay of the recorded Isaac Gym rollout (trace.npz) with the LLM schedule overlaid",
             fontsize=9, color=MUTED)

    used = {t[k] for t in scene["tasks"] for k in ("object_id", "support_object_id") if k in t}
    for o in scene["initial_observation"]["objects"]:
        if o["object_id"] not in used:
            ax.add_patch(Polygon(rect(o["position_m"], o["dimensions_m"], o["orientation_xyzw"]),
                                 fill=False, ec="#c4c3bc", lw=0.7, ls="--", zorder=1))
    goals = {tuple(t["object_goal_position_m"]) for t in scene["tasks"] if "object_goal_position_m" in t}
    for g in goals:
        ax.plot(g[0], g[1], marker="*", ms=13, color=INK, mec="white", mew=0.8, zorder=2)

    polys = {k: ax.add_patch(Polygon(rect(boxes[0, k, :3], sizes[k], boxes[0, k, 3:]), fc="#e6e4dc",
                                     ec="#5c5b56", lw=0.9, zorder=3)) for k in slots}
    paths = [ax.plot([], [], lw=1.0, color=c, alpha=0.45, zorder=4)[0] for c in color]
    dots = [ax.plot([], [], "o", ms=9, color=c, mec=c, mew=1.6, zorder=6)[0] for c in color]
    labels = [ax.text(0, 0, "", fontsize=8, color=INK, zorder=7) for _ in ids]
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=12, color=INK, va="top", family="monospace")

    pts = [roots[..., :2].reshape(-1, 2)] + [boxes[:, k, :2] for k in slots] + [np.array([g[:2] for g in goals])]
    pts = np.concatenate([p for p in pts if len(p)])
    lo, hi = pts.min(0) - 2.0, pts.max(0) + 2.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(color="#ecebe6", lw=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d6d5ce")
    key.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=c, label=r) for r, c in ROLE_COLOR.items()]
               + [Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=INK, label="frozen (waiting)"),
                  Line2D([], [], marker="X", ls="", ms=8, color=INK, label="fallen"),
                  Line2D([], [], marker="*", ls="", ms=11, color=INK, label="object goal"),
                  Patch(fill=False, ec="#c4c3bc", ls="--", label="distractor (observed only)"),
                  Patch(color=FROZEN, label="frozen until LLM release"),
                  Line2D([], [], marker="D", ls="", mfc="white", mec=INK, label="LLM predicted event")],
               loc="upper left", ncol=3, fontsize=7.5, frameon=False)

    horizon = T * dt
    predicted = {}
    for ev in d["response"]["predicted_episode_events"]:
        predicted.setdefault(ev["agent_id"], []).append(ev["episode_time_s"])
    for i, aid in enumerate(ids):
        r = release[i] * dt
        tl.barh(i, r, left=0, height=0.55, color=FROZEN, zorder=1)
        tl.plot([r, horizon], [i, i], lw=2.2, color=color[i], solid_capstyle="butt", zorder=2)
        tl.plot(predicted.get(aid, []), [i] * len(predicted.get(aid, [])), "D", ms=6, mfc="white", mec=INK, mew=1.2, zorder=3)
        if d["fall"][i] >= 0:
            tl.plot(d["fall"][i] * dt, i, "X", ms=8, color=INK, zorder=4)
    tl.set_yticks(range(N), [f"{a} {tasks[a]['task_label']}" for a in ids], fontsize=7 if N > 8 else 8)
    tl.set_ylim(N - 0.4, -0.6)
    tl.set_xlim(0, horizon + 2.5)
    tl.set_xlabel("simulation time (s)", fontsize=8, color=MUTED)
    tl.tick_params(axis="x", labelsize=7, colors=MUTED)
    tl.grid(axis="x", color="#ecebe6", lw=0.6)
    for s in tl.spines.values():
        s.set_color("#d6d5ce")
    tl.set_title("release schedule  (bar = frozen, line = running)", fontsize=9, color=INK, loc="left")
    cursor = tl.axvline(0, color=INK, lw=1.0, zorder=5)
    results = [tl.text(horizon + 0.6, i, "", fontsize=10, va="center", color=INK) for i in range(N)]

    info_text = info.text(0, 1, "", fontsize=9, color=INK, va="top", family="monospace", transform=info.transAxes)
    waiting_note = "outcome is shown when the 30 s rollout ends"
    final = ["rollout outcome (recorded):"] + outcome_lines(scene, d, index)
    final.append("task success (region, ever): " + " ".join(
        f"{a}{'✓' if d['ever'][i] else '✗'}" for i, a in enumerate(ids)))

    def update(f):
        t = min(f, T - 1)
        for k, poly in polys.items():
            poly.set_xy(rect(boxes[t, k, :3], sizes[k], boxes[t, k, 3:]))
        for i, aid in enumerate(ids):
            x, y = roots[t, i, :2]
            frozen = t < release[i]
            fallen = 0 <= d["fall"][i] <= t
            paths[i].set_data(roots[:t + 1, i, 0], roots[:t + 1, i, 1])
            dots[i].set_data([x], [y])
            dots[i].set_marker("X" if fallen else "o")
            dots[i].set_markerfacecolor("white" if frozen else color[i])
            labels[i].set_position((x + 0.3, y + 0.3))
            labels[i].set_text(f"{aid} wait {(release[i] - t) * dt:.1f}s" if frozen else aid)
        clock.set_text(f"t = {t * dt:5.1f} s")
        cursor.set_xdata([t * dt, t * dt])
        done = f >= T - 1
        for i in range(N):
            results[i].set_text(("✓" if d["ever"][i] else "✗") if done else "")
        info_text.set_text("\n".join(final if done else [waiting_note]))

    VIDEO.mkdir(exist_ok=True)
    path = VIDEO / f"{SCENES.index(name) + 1:02d}_{setting}_{scene['scene_id']}.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with writer.saving(fig, str(path), dpi=100):
        for f in range(T + HOLD):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    return path


if __name__ == "__main__":
    names = [SCENES[int(i) - 1] for i in sys.argv[1:]] or SCENES
    with Pool(min(len(names), 10)) as pool:
        for p in pool.imap_unordered(render, names):
            print(p.name, flush=True)
