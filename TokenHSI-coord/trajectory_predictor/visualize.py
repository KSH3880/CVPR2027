"""Isaac-Gym-free 2D animation of oracle and predicted joint trajectories."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FFMpegWriter
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .dataset import TrajectoryDataset
from .geometry import resample_coarse
from .oracle import timed_clearance, timed_positions
from .schema import AGENTS, SPEED_VALUES, state_from_batch


AGENT_COLORS = ("#2878B5", "#E87500")


def _first_batch(dataset: TrajectoryDataset, scenes: int):
    if len(dataset) == 0:
        raise ValueError("visualization dataset has no oracle-valid scenes")
    count = min(int(scenes), len(dataset))
    return next(iter(DataLoader(dataset, batch_size=count, shuffle=False)))


@torch.no_grad()
def render_checkpoint_video(
    checkpoint: Union[str, Path],
    data: Union[str, Path],
    output: Union[str, Path],
    *,
    scenes: int = 4,
    fps: int = 10,
    max_frames: int = 120,
) -> Path:
    """Render fixed held-out scenes; all inference and drawing stays on CPU."""
    if scenes <= 0 or fps <= 0 or max_frames <= 1:
        raise ValueError("scenes/fps must be positive and max_frames must exceed one")
    model, payload = load_checkpoint(checkpoint, "cpu")
    dataset = TrajectoryDataset(data, augment=False, valid_only=True)
    batch = _first_batch(dataset, scenes)
    state = state_from_batch(batch)
    prediction = model(state)
    pred_coarse = prediction["coarse_world"]
    pred_speed = prediction["speed_logits"].argmax(dim=-1)
    gt_coarse = batch["coarse_path"]
    gt_speed = batch["speed_class"].long()
    pred_dense, pred_end = resample_coarse(pred_coarse, with_end=True)
    gt_dense, gt_end = resample_coarse(gt_coarse, with_end=True)
    count = state.batch_size

    pred_pos, _, _ = timed_positions(
        pred_dense.reshape(count * AGENTS, -1, 2),
        pred_end.reshape(-1),
        pred_speed.reshape(count * AGENTS, 4),
    )
    pred_pos = pred_pos.reshape(count, AGENTS, -1, 2)
    pred_clear, _, _, _ = timed_clearance(
        pred_dense[:, 0], pred_dense[:, 1], pred_end[:, 0], pred_end[:, 1],
        pred_speed[:, 0], pred_speed[:, 1],
    )
    frame_index = torch.linspace(
        0, pred_pos.shape[2] - 1, min(max_frames, pred_pos.shape[2])
    ).round().long()

    cols = min(2, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 6.4 * rows), squeeze=False)
    moving = []
    for scene in range(rows * cols):
        ax = axes.flat[scene]
        if scene >= count:
            ax.axis("off")
            continue
        all_points = torch.cat((pred_coarse[scene].reshape(-1, 2), gt_coarse[scene].reshape(-1, 2)))
        low = all_points.amin(dim=0) - 0.8
        high = all_points.amax(dim=0) + 0.8
        ax.set_xlim(float(low[0]), float(high[0]))
        ax.set_ylim(float(low[1]), float(high[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        scenario = "Cross" if int(batch["scenario"][scene]) == 1 else "Free"
        speed_text = " | ".join(
            f"A{agent}:" + "/".join(f"{SPEED_VALUES[index]:.3g}" for index in pred_speed[scene, agent])
            for agent in range(AGENTS)
        )
        ax.set_title(
            f"{scenario} #{scene}  predicted clearance={float(pred_clear[scene]):.2f}m\n{speed_text}",
            fontsize=10,
        )
        for agent, color in enumerate(AGENT_COLORS):
            gt_last = int((gt_end[scene, agent] / 0.1).round().clamp(1, gt_dense.shape[2] - 1))
            pred_last = int((pred_end[scene, agent] / 0.1).round().clamp(1, pred_dense.shape[2] - 1))
            ax.plot(
                gt_dense[scene, agent, :gt_last + 1, 0],
                gt_dense[scene, agent, :gt_last + 1, 1],
                linestyle="--", linewidth=2.0, color=color, alpha=0.45,
                label=f"A{agent} oracle",
            )
            ax.plot(
                pred_dense[scene, agent, :pred_last + 1, 0],
                pred_dense[scene, agent, :pred_last + 1, 1],
                linewidth=2.3, color=color, label=f"A{agent} predicted",
            )
            ax.scatter(*state.root_xy[scene, agent], marker="o", s=45, color=color, edgecolor="black")
            ax.scatter(*state.box_xyz[scene, agent, :2], marker="s", s=65, color=color, edgecolor="black")
            ax.scatter(*state.goal_xy[scene, agent], marker="*", s=125, color=color, edgecolor="black")
            dot, = ax.plot([], [], marker="o", markersize=11, color=color, markeredgecolor="white")
            moving.append((scene, agent, dot))
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Joint trajectory predictor — epoch {payload.get('epoch', '?')}\n"
        "dashed=oracle, solid=prediction, square=box, star=goal",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    writer = FFMpegWriter(fps=fps, metadata={"title": output.stem}, bitrate=1800)
    try:
        with writer.saving(fig, str(temporary), dpi=110):
            for index in frame_index:
                for scene, agent, dot in moving:
                    point = pred_pos[scene, agent, index]
                    dot.set_data([float(point[0])], [float(point[1])])
                writer.grab_frame()
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return output
