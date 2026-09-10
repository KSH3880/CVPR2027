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


STACK_SCHEMA_VERSION = "tokenhsi-stack-planner-v1"
STACK_CANDIDATES = 4

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
    "STACK_SCHEMA_VERSION",
]

