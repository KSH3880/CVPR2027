"""Small, stable TensorBoard dashboard for Carry planner training."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping


# Keep the event file focused; metrics.jsonl remains the complete raw record.
SCALARS = {
    "Reward/total": "reward",
    "Reward/executor": "base_reward",
    "Reward/progress": "progress",
    "Reward/done_rate": "done_rate",
    "Collision/actual_ratio": "collision_ratio",
    "Collision/actual_cost": "collision_cost",
    "Collision/agent_agent_cost": "collision_agent_agent_cost",
    "Collision/agent_box_cost": "collision_agent_box_cost",
    "Collision/box_box_cost": "collision_box_box_cost",
    "Collision/analytic_loss": "analytic_collision_loss",
    "Plan/sample_valid_fraction": "sample_plan_valid_fraction",
    "Plan/mean_valid_fraction": "mean_plan_valid_fraction",
    "Plan/sample_curve_fraction": "sample_plan_curve_fraction",
    "Plan/mean_curve_fraction": "mean_plan_curve_fraction",
    "Plan/invalid_penalty": "invalid_plan_penalty",
    "Plan/sample_max_turn_deg": "sample_plan_max_turn_deg",
    "Plan/mean_max_turn_deg": "mean_plan_max_turn_deg",
    "Path/sample_deviation_m": "sample_path_deviation",
    "Path/mean_deviation_m": "mean_path_deviation",
    "Path/future_excess_m": "mean_future_excess_m",
    "Path/replan_displacement_m": "mean_replan_displacement",
    "Path/direction_error_deg": "mean_direction_error_deg",
    "Loss/policy": "policy_loss",
    "Loss/value": "value_loss",
    "Loss/path_smoothness": "path_smoothness_loss",
    "Loss/analytic_curvature": "analytic_curvature_loss",
    "Loss/weighted_excess_length": "weighted_excess_length_loss",
    "Exploration/path_std": "path_delta_std",
}


def write_scalars(writer, metrics: Mapping[str, object]) -> int:
    """Write selected finite scalars at their original training iteration."""
    step = int(metrics["iteration"])
    count = 0
    for tag, key in SCALARS.items():
        if key not in metrics:
            continue
        value = float(metrics[key])
        if math.isfinite(value):
            writer.add_scalar(tag, value, global_step=step)
            count += 1
    return count


def make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard가 설치되지 않았습니다: python -m pip install tensorboard"
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def import_jsonl(metrics_path: Path, writer, *, follow: bool = False,
                 poll_seconds: float = 2.0) -> int:
    """Import existing JSONL; optionally keep following an active training run."""
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    imported = 0
    with metrics_path.open("r", encoding="utf-8") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                if not follow:
                    break
                time.sleep(poll_seconds)
                continue
            if not line.endswith("\n"):
                stream.seek(position)
                if not follow:
                    break
                time.sleep(poll_seconds)
                continue
            metrics = json.loads(line)
            write_scalars(writer, metrics)
            writer.flush()
            imported += 1
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="runs/carry_planner/<tag>")
    parser.add_argument("--follow", action="store_true", help="계속 새 iteration 읽기")
    parser.add_argument("--log-dir", type=Path, help="event 출력 위치 (기본: run_dir/tensorboard_imported)")
    args = parser.parse_args()
    metrics_path = args.run_dir / "metrics.jsonl"
    # Separate old-run imports from event files produced by new training.
    log_dir = args.log_dir or args.run_dir / "tensorboard_imported"
    if not metrics_path.is_file():
        parser.error(f"metrics.jsonl 없음: {metrics_path}")
    if log_dir.is_dir() and any(log_dir.glob("events.out.tfevents.*")):
        parser.error(f"기존 TensorBoard event가 있음: {log_dir}; 새 --log-dir를 지정하세요")
    writer = make_writer(log_dir)
    try:
        count = import_jsonl(metrics_path, writer, follow=args.follow)
        print(f"imported {count} iterations to {log_dir}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
