"""Measure frozen ms18 speed, pickup and crossing-time response.

The coordinator is not trained here.  A deterministic analytic path with six
fixed command profiles is installed every replan interval, while actual
executor state and humanoid body clearance are recorded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Isaac Gym requires this import before torch.
from isaacgym import gymapi as _gymapi  # noqa: F401

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENHSI_ROOT = REPO_ROOT / "tokenhsi"
if str(TOKENHSI_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENHSI_ROOT))

import run as tokenhsi_run  # noqa: E402
from coordinator.executor_calibration import (  # noqa: E402
    PROFILE_NAMES,
    requested_speeds,
    summarize_measurement,
)
from coordinator.planner import crossing_arrival_metrics  # noqa: E402
from coordinator.schema import AGENTS, PATH_POINTS  # noqa: E402
from coordinator.train_closed_loop import _make_player, _refresh_obs  # noqa: E402
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _atomic_json(path: Path, value: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _initial_crossing(path: torch.Tensor, speed: torch.Tensor, dwell: torch.Tensor):
    metrics = crossing_arrival_metrics(
        path[:, None], speed[:, None], dwell[:, None]
    )
    row = torch.arange(path.shape[0], device=path.device)
    index0 = metrics["crossing_index0"][:, 0]
    index1 = metrics["crossing_index1"][:, 0]
    # crossing_arrival_metrics uses a 34-point timeline with P16 duplicated
    # across pickup dwell. Map that timeline back to the public 33-point path.
    path_index0 = torch.where(index0 <= 16, index0, index0 - 1)
    path_index1 = torch.where(index1 <= 16, index1, index1 - 1)
    points = torch.stack(
        (path[row, 0, path_index0], path[row, 1, path_index1]), dim=1
    )
    predicted = torch.stack(
        (
            metrics["crossing_arrival_t0"][:, 0],
            metrics["crossing_arrival_t1"][:, 0],
        ),
        dim=1,
    )
    return points, predicted


@torch.no_grad()
def run_measurement(player, output_dir: Path, steps: int, replan_steps: int) -> Dict[str, object]:
    task = player.env.task
    envs = int(task.num_envs)
    device = torch.device(task.device)
    if envs < len(PROFILE_NAMES):
        raise ValueError(f"measurement needs at least {len(PROFILE_NAMES)} envs")
    if steps <= 0 or replan_steps <= 0:
        raise ValueError("steps and replan_steps must be positive")

    player.env.reset()
    state0 = task.coord_state()
    profile_ids = torch.arange(envs, device=device) % len(PROFILE_NAMES)
    held_age = torch.full((envs, AGENTS), -1, device=device, dtype=torch.long)
    requested = requested_speeds(profile_ids, held_age)
    path0, _ = task._analytic_plan(state0)
    speed0 = requested[:, :, None].expand(-1, -1, PATH_POINTS).clone()
    dwell0 = torch.where(
        state0.held >= 0.5,
        torch.zeros_like(state0.held),
        torch.full_like(state0.held, 1.5),
    )
    crossing_points, predicted_crossing_time = _initial_crossing(
        path0, speed0, dwell0
    )

    actual_crossing_distance = torch.full(
        (envs, AGENTS), float("inf"), device=device
    )
    actual_crossing_time = torch.full((envs, AGENTS), -1.0, device=device)
    pickup_near_step = torch.full((envs, AGENTS), -1, device=device, dtype=torch.long)
    pickup_held_step = torch.full_like(pickup_near_step, -1)
    active = torch.ones(envs, device=device, dtype=torch.bool)
    body_episode_min = torch.full((envs,), float("inf"), device=device)

    records: Dict[str, List[np.ndarray]] = {
        "active": [],
        "requested_speed": [],
        "sent_speed": [],
        "root_velocity_speed": [],
        "displacement_speed": [],
        "held": [],
        "phase": [],
        "root_xy": [],
        "box_xyz": [],
        "goal_xy": [],
        "root_box_distance": [],
        "body_min_distance": [],
        "root_distance": [],
        "box_distance": [],
        "human_box_distance": [],
    }

    obs = _refresh_obs(player)
    player.get_batch_size(obs, 1)
    executed_steps = 0
    for step in range(steps):
        before = task.coord_state()
        if step % replan_steps == 0:
            requested = requested_speeds(profile_ids, held_age)
            path, _ = task._analytic_plan(before)
            speed = requested[:, :, None].expand(-1, -1, PATH_POINTS).clone()
            dwell = torch.where(
                before.held >= 0.5,
                torch.zeros_like(before.held),
                torch.full_like(before.held, 1.5),
            )
            output = {
                "path_world": path[:, None],
                "speed": speed[:, None],
                "pickup_dwell": dwell[:, None],
                # Runtime diagnostics share the learned-output contract even
                # though this measurement deliberately installs analytic paths.
                "control_residual": torch.zeros(
                    envs, 1, AGENTS, 4, 2,
                    device=device, dtype=path.dtype,
                ),
            }
            task.install_external_coord(output)

        obs = _refresh_obs(player)
        sent = task._coord_cmd_speed.reshape(envs, AGENTS).clone()
        root_before = before.root_xy.clone()
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, _, done_rows, _ = player.env.step(action)
        after = task.coord_state()

        root_box = (after.root_xy - after.box_xyz[..., :2]).norm(dim=-1)
        root_distance = (after.root_xy[:, 0] - after.root_xy[:, 1]).norm(dim=-1)
        box_distance = (
            after.box_xyz[:, 0, :2] - after.box_xyz[:, 1, :2]
        ).norm(dim=-1)
        hb01 = (after.root_xy[:, 0] - after.box_xyz[:, 1, :2]).norm(dim=-1)
        hb10 = (after.root_xy[:, 1] - after.box_xyz[:, 0, :2]).norm(dim=-1)
        human_box = torch.minimum(hb01, hb10)
        body_distance = task.agent_min_dist().reshape(envs, AGENTS).amin(dim=1)
        displacement_speed = (after.root_xy - root_before).norm(dim=-1) / float(task.dt)
        velocity_speed = after.root_vel_xy.norm(dim=-1)

        now_s = torch.full(
            (envs, AGENTS), (step + 1) * float(task.dt),
            device=device, dtype=after.root_xy.dtype,
        )
        crossing_distance = (after.root_xy - crossing_points).norm(dim=-1)
        improve = active[:, None] & (crossing_distance < actual_crossing_distance)
        actual_crossing_distance = torch.where(
            improve, crossing_distance, actual_crossing_distance
        )
        actual_crossing_time = torch.where(improve, now_s, actual_crossing_time)
        body_episode_min = torch.where(
            active, torch.minimum(body_episode_min, body_distance), body_episode_min
        )

        enter_near = (
            (pickup_near_step < 0) & (root_box <= 0.7) & (after.held < 0.5)
            & active[:, None]
        )
        pickup_near_step = torch.where(
            enter_near, torch.full_like(pickup_near_step, step + 1), pickup_near_step
        )
        first_held = (
            (pickup_held_step < 0) & (after.held >= 0.5) & active[:, None]
        )
        pickup_held_step = torch.where(
            first_held, torch.full_like(pickup_held_step, step + 1), pickup_held_step
        )
        held_age = torch.where(
            after.held >= 0.5, torch.clamp(held_age + 1, min=0),
            torch.full_like(held_age, -1),
        )

        records["active"].append(_cpu(active))
        records["requested_speed"].append(_cpu(requested))
        records["sent_speed"].append(_cpu(sent))
        records["root_velocity_speed"].append(_cpu(velocity_speed))
        records["displacement_speed"].append(_cpu(displacement_speed))
        records["held"].append(_cpu(after.held >= 0.5))
        records["phase"].append(_cpu(after.phase))
        records["root_xy"].append(_cpu(after.root_xy))
        records["box_xyz"].append(_cpu(after.box_xyz))
        records["goal_xy"].append(_cpu(after.goal_xy))
        records["root_box_distance"].append(_cpu(root_box))
        records["body_min_distance"].append(_cpu(body_distance))
        records["root_distance"].append(_cpu(root_distance))
        records["box_distance"].append(_cpu(box_distance))
        records["human_box_distance"].append(_cpu(human_box))

        done_env = done_rows.reshape(envs, AGENTS).any(dim=1)
        active &= ~done_env
        executed_steps = step + 1
        if not bool(active.any()):
            break

    pickup_dwell = torch.where(
        (pickup_near_step >= 0) & (pickup_held_step >= pickup_near_step),
        (pickup_held_step - pickup_near_step).to(torch.float32) * float(task.dt),
        torch.full((envs, AGENTS), -1.0, device=device),
    )
    arrays = {key: np.stack(value, axis=0) for key, value in records.items()}
    arrays.update(
        {
            "profile_id": _cpu(profile_ids),
            "pickup_dwell_s": _cpu(pickup_dwell),
            "predicted_crossing_time": _cpu(predicted_crossing_time),
            "actual_crossing_time": _cpu(actual_crossing_time),
            "actual_crossing_distance": _cpu(actual_crossing_distance),
            "body_episode_min_distance": _cpu(body_episode_min),
            "crossing_point": _cpu(crossing_points),
        }
    )
    metadata: Dict[str, object] = {
        "scenario": os.environ.get("MS_SCEN", "free"),
        "seed": int(os.environ.get("MS_SEED", "0")),
        "envs": envs,
        "requested_steps": steps,
        "executed_steps": executed_steps,
        "replan_steps": replan_steps,
        "dt": float(task.dt),
        "profile_names": list(PROFILE_NAMES),
        "executor_checkpoint": os.environ.get("MS_MEASURE_EXEC_CKPT", ""),
        "fixed_prediction_pickup_dwell_s": 1.5,
        "body_collision_threshold_m": 0.3,
    }
    summary = summarize_measurement(arrays, metadata)
    _atomic_npz(output_dir / "raw.npz", arrays)
    _atomic_json(output_dir / "summary.json", summary)
    print(
        "MS18_MEASURE_SUMMARY "
        f"scenario={metadata['scenario']} envs={envs} steps={executed_steps} "
        f"pickup_median={summary['pickup_dwell_s']['median']:.4f} "
        f"approach_alpha={summary['response']['approach']['alpha']:.4f} "
        f"carry_alpha={summary['response']['carry']['alpha']:.4f} "
        f"collision_lt03={summary['collision']['fraction_below_0p3']:.4f} "
        f"crossing_abs_error={summary['crossing']['arrival_abs_error_s']['median']:.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    set_np_formatting()
    args = get_args()
    os.environ["COORD_PROVIDER"] = "external"
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(
        cfg_train["params"].get("seed", 0),
        cfg_train["params"].get("torch_deterministic", False),
    )
    cfg_train["params"]["seed"] = seed
    cfg_train["params"]["config"]["seed"] = seed
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    output_dir_raw = os.environ.get("MS_MEASURE_OUT")
    if not output_dir_raw:
        raise ValueError("MS_MEASURE_OUT is required")
    output_dir = Path(output_dir_raw).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    player = _make_player(args, cfg, cfg_train)
    run_measurement(
        player,
        output_dir,
        _env_int("MS_MEASURE_STEPS", 600),
        _env_int("MS_MEASURE_REPLAN_STEPS", 6),
    )


if __name__ == "__main__":
    main()
