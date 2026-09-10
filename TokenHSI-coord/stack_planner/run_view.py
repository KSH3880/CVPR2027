"""Register only the isolated stack-planner viewer task, then run TokenHSI."""

from __future__ import annotations

import sys
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401

COORD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = COORD_ROOT.parent
MASTEER_ROOT = WORKSPACE / "TokenHSI-masteer"
sys.path.insert(0, str(MASTEER_ROOT / "tokenhsi"))
sys.path.insert(0, str(COORD_ROOT))

import utils.parse_task as task_registry  # noqa: E402
from stack_planner.view_env import HumanoidMAStackPlannerView  # noqa: E402

task_registry.HumanoidMAStackPlannerView = HumanoidMAStackPlannerView

from run import main  # noqa: E402


if __name__ == "__main__":
    main()
