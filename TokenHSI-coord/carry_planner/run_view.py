"""Register the plain-Carry planner viewer task and run TokenHSI."""

from __future__ import annotations

import sys
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401

COORD_ROOT = Path(__file__).resolve().parents[1]
TOKENHSI_ROOT = COORD_ROOT / "tokenhsi"
sys.path.insert(0, str(TOKENHSI_ROOT))
sys.path.insert(0, str(COORD_ROOT))

import utils.parse_task as task_registry  # noqa: E402
from carry_planner.view_env import HumanoidMACarryPlannerView  # noqa: E402

task_registry.HumanoidMACarryPlannerView = HumanoidMACarryPlannerView

from run import main  # noqa: E402


if __name__ == "__main__":
    main()
