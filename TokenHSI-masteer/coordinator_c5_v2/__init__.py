"""Standalone role-conditioned C5 V2 trajectory proposal package."""

from .core import (
    ACTION_DIM,
    MANEUVER_COUNT,
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
    MANEUVER_STRAIGHT,
    ROLE_COUNT,
    C5Config,
    C5ProposalActor,
    C5TrajectoryCodec,
    canonical_action,
    encode_state,
)
from .teacher import CEMConfig, CEMTeacher, TeacherResult, oracle_summary

__all__ = [
    "ACTION_DIM",
    "ROLE_COUNT",
    "MANEUVER_COUNT",
    "MANEUVER_STRAIGHT",
    "MANEUVER_LEFT",
    "MANEUVER_RIGHT",
    "MANEUVER_SLOW",
    "C5Config",
    "C5ProposalActor",
    "C5TrajectoryCodec",
    "canonical_action",
    "encode_state",
    "CEMConfig",
    "CEMTeacher",
    "TeacherResult",
    "oracle_summary",
]
