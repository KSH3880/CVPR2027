"""Create a deterministic, untrained sparse-spline planner for viewer checks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from coordinator.schema import MAX_SPEED, MIN_SPEED
from stack_planner.checkpoint import save_stack_checkpoint
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner
from stack_planner.schema import AGENTS, CARRY_LEARNED_PATH_POINTS, CARRY_PATH_KNOTS


def _speed_logit(speed: torch.Tensor) -> torch.Tensor:
    unit = ((speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).clamp(1e-5, 1.0 - 1e-5)
    return torch.log(unit / (1.0 - unit))


def build_sanity_planner(control_scale: float = 4.0) -> StackTrajectoryPlanner:
    """Return an untrained planner with fixed, interpretable detour outputs."""
    planner = StackTrajectoryPlanner(StackPlannerConfig(
        candidates=1,
        history_steps=4,
        delta_scale=1.0,
        retreat_delta_scale=1.0,
        path_update_alpha=0.5,
        plain_carry=True,
        carry_control_scale=control_scale,
    ))

    # Default Cross layout: A1 travels mostly along +X and bends toward +Y;
    # A2 travels mostly along +Y and bends toward -X. The four rows are the
    # learned approach point followed by three learned carry points.
    offset = torch.tensor([
        [[0.0, 0.55], [0.0, 1.00], [0.0, 1.00], [0.0, 0.55]],
        [[-0.55, 0.0], [-1.00, 0.0], [-1.00, 0.0], [-0.55, 0.0]],
    ])
    if offset.shape != (AGENTS, CARRY_LEARNED_PATH_POINTS, 2):
        raise AssertionError("sanity path profile has the wrong shape")
    raw_offset = torch.atanh((offset / control_scale).clamp(-0.999, 0.999))

    # Slow both agents near pickup and the middle of the carry leg, with a
    # deliberately different profile per agent so speed ribbon changes are
    # visually obvious.
    speed = torch.tensor([
        [1.20, 1.05, 0.70, 1.10, 0.60, 1.05, 0.50],
        [1.00, 0.90, 0.65, 0.85, 0.50, 1.00, 0.45],
    ])
    if speed.shape != (AGENTS, CARRY_PATH_KNOTS):
        raise AssertionError("sanity speed profile has the wrong shape")

    with torch.no_grad():
        path_final = planner.heads.paths[0][-1]
        path_final.weight.zero_()
        path_final.bias.copy_(raw_offset.reshape(-1))
        speed_final = planner.heads.speeds[0][-1]
        speed_final.weight.zero_()
        speed_final.bias.copy_(_speed_logit(speed).reshape(-1))
    return planner.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--control-scale", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.control_scale) or args.control_scale <= 1.0:
        raise SystemExit("--control-scale must be finite and greater than 1m")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing checkpoint: {output}")
    planner = build_sanity_planner(args.control_scale)
    save_stack_checkpoint(
        output, planner, step=0,
        metrics={"sanity_untrained": 1.0},
        extras={
            "sanity_profile": "opposite_detours_with_speed_dips",
            "trained": False,
        },
    )
    print(f"wrote untrained sanity checkpoint: {output}")


if __name__ == "__main__":
    main()
