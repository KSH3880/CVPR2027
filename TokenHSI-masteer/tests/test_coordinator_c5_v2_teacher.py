import unittest

import torch

from coordinator.schema import CoordinatorState
from coordinator_c5_v2 import (
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    CEMConfig,
    CEMTeacher,
    oracle_summary,
)


def make_carried_crossing(batch=2):
    root = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]]).expand(batch, -1, -1).clone()
    goal = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]).expand(batch, -1, -1).clone()
    return CoordinatorState(
        root_xy=root, heading=torch.zeros(batch, 2), root_vel_xy=torch.zeros(batch, 2, 2),
        box_xyz=torch.cat((root, torch.full((batch, 2, 1), 0.8)), dim=-1),
        box_heading=torch.zeros(batch, 2), box_vel_xy=torch.zeros(batch, 2, 2),
        box_size_xy=torch.full((batch, 2, 2), 0.5), goal_xy=goal,
        held=torch.ones(batch, 2), phase=torch.full((batch, 2), 2.0),
    )


class C5V2TeacherTest(unittest.TestCase):
    def test_cem_teacher_contract_and_mode_labels(self):
        state = make_carried_crossing()
        teacher = CEMTeacher(config=CEMConfig(population=12, elites=3, iterations=2))
        role = torch.tensor([0, 1])
        maneuver = torch.tensor([MANEUVER_LEFT, MANEUVER_RIGHT])
        result = teacher.search(
            state, role, maneuver, generator=torch.Generator().manual_seed(7)
        )
        self.assertEqual(result.action.shape, (2, 3, 48))
        self.assertEqual(result.path_world.shape, (2, 3, 2, 33, 2))
        self.assertEqual(result.speed.shape, (2, 3, 2, 33))
        torch.testing.assert_close(result.role, role)
        torch.testing.assert_close(result.maneuver, maneuver)
        self.assertTrue(torch.isfinite(result.score).all())

    def test_oracle_unresolved_is_separate_from_invalid(self):
        state = make_carried_crossing()
        teacher = CEMTeacher(config=CEMConfig(population=8, elites=2, iterations=1))
        role = torch.tensor([0, 1])
        left = teacher.search(
            state, role, torch.full_like(role, MANEUVER_LEFT),
            generator=torch.Generator().manual_seed(1),
        )
        right = teacher.search(
            state, role, torch.full_like(role, MANEUVER_RIGHT),
            generator=torch.Generator().manual_seed(2),
        )
        summary = oracle_summary([left, right])
        self.assertEqual(summary["scenarios"], 2)
        self.assertEqual(
            summary["oracle_safe"] + summary["unresolved_nonstop"], 2
        )
        self.assertEqual(
            summary["valid_but_unsafe"] + summary["no_valid_plan"],
            summary["unresolved_nonstop"],
        )


if __name__ == "__main__":
    unittest.main()
