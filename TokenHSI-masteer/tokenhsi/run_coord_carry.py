"""Dedicated entry point for the original simultaneous coordinator executor."""

import sys
from pathlib import Path

# Isaac Gym must be imported before torch (also through coordinator modules).
from isaacgym import gymapi  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))
sys.path.insert(0, str(ROOT / "TokenHSI-coord"))

import utils.parse_task as task_registry
from env.tasks.adapt_interaction_skills.humanoid_ma_coord_carry import (
    HumanoidMACoordCarry,
)

task_registry.HumanoidMACoordCarry = HumanoidMACoordCarry

from run import main

if __name__ == "__main__":
    main()
