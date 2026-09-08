"""Dedicated entry point: register the new task only in this process."""

import sys
from pathlib import Path

# Isaac Gym must precede torch, including transitively imported coordinator code.
from isaacgym import gymapi  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))
sys.path.insert(0, str(ROOT / "TokenHSI-coord"))

import utils.parse_task as task_registry
from env.tasks.adapt_interaction_skills.humanoid_ma_coord_sequential_stack import (
    HumanoidMACoordSequentialStack,
)

task_registry.HumanoidMACoordSequentialStack = HumanoidMACoordSequentialStack

from run import main

if __name__ == "__main__":
    main()
