"""Register the plain-Carry planner viewer task and run TokenHSI."""

from __future__ import annotations

import sys
import os
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401

COORD_ROOT = Path(__file__).resolve().parents[1]
TOKENHSI_ROOT = COORD_ROOT / "tokenhsi"
sys.path.insert(0, str(TOKENHSI_ROOT))
sys.path.insert(0, str(COORD_ROOT))

import utils.parse_task as task_registry  # noqa: E402
from carry_planner.view_env import HumanoidMACarryPlannerView  # noqa: E402

task_registry.HumanoidMACarryPlannerView = HumanoidMACarryPlannerView

import run as tokenhsi_run  # noqa: E402


def main():
    """Run an effectively unbounded interactive player, not finite eval."""
    games = int(os.environ.get("CARRY_PLANNER_VIEW_GAMES", "1000000000"))
    if games < 1:
        raise ValueError("CARRY_PLANNER_VIEW_GAMES must be positive")
    original_load_cfg = tokenhsi_run.load_cfg

    def load_view_cfg(args):
        cfg, cfg_train, logdir = original_load_cfg(args)
        player = cfg_train["params"]["config"].setdefault("player", {})
        player["games_num"] = games
        return cfg, cfg_train, logdir

    tokenhsi_run.load_cfg = load_view_cfg
    try:
        tokenhsi_run.main()
    finally:
        tokenhsi_run.load_cfg = original_load_cfg


if __name__ == "__main__":
    main()
