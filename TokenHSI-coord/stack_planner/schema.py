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


STACK_SCHEMA_VERSION = "tokenhsi-stack-planner-v12"
STACK_CANDIDATES = 4
STACK_PATH_POINTS = PATH_POINTS
STACK_PATH_DELTA_DIM = AGENTS * (STACK_PATH_POINTS - 1) * 2
STACK_PATH_INPUT_DIM = AGENTS * STACK_PATH_POINTS * 2

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
    "STACK_SCHEMA_VERSION",
]
