"""Stack-planner contract isolated from the existing coordinator checkpoints."""

from coordinator.schema import (  # re-export the unchanged public input
    ACCEL_KNOTS,
    AGENTS,
    MAX_ACCEL,
    MAX_SPEED,
    MIN_SPEED,
    PATH_DS,
    PATH_POINTS,
    PATH_VERTICES,
    STATE_KEYS,
    STEER_HORIZON_SECONDS,
    STEER_POINTS,
    CoordinatorState,
)


STACK_SCHEMA_VERSION = "tokenhsi-stack-planner-v16"
# Establish the path planner with a single policy head first. Multi-head
# candidate evaluation remains available through an explicit config override.
STACK_CANDIDATES = 1
STACK_PATH_POINTS = PATH_POINTS
STACK_PATH_DELTA_DIM = AGENTS * (STACK_PATH_POINTS - 1) * 2
STACK_PATH_INPUT_DIM = AGENTS * STACK_PATH_POINTS * 2
STACK_SPEED_DIM = AGENTS * STACK_PATH_POINTS
CARRY_LEARNED_PATH_POINTS = 4
CARRY_PATH_KNOTS = 7
CARRY_PATH_DIM = AGENTS * CARRY_LEARNED_PATH_POINTS * 2
CARRY_SPEED_DIM = AGENTS * CARRY_PATH_KNOTS

__all__ = [
    "ACCEL_KNOTS",
    "AGENTS",
    "MAX_ACCEL",
    "MAX_SPEED",
    "MIN_SPEED",
    "PATH_DS",
    "PATH_POINTS",
    "PATH_VERTICES",
    "STATE_KEYS",
    "STEER_HORIZON_SECONDS",
    "STEER_POINTS",
    "CoordinatorState",
    "STACK_CANDIDATES",
    "STACK_PATH_POINTS",
    "STACK_PATH_DELTA_DIM",
    "STACK_PATH_INPUT_DIM",
    "STACK_SPEED_DIM",
    "CARRY_LEARNED_PATH_POINTS",
    "CARRY_PATH_KNOTS",
    "CARRY_PATH_DIM",
    "CARRY_SPEED_DIM",
    "STACK_SCHEMA_VERSION",
]
