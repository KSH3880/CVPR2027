from __future__ import annotations

import unittest

import torch

from carry_planner.layout import converging_goal_xy


class ConvergingGoalLayoutTest(unittest.TestCase):
    def test_goals_share_center_and_preserve_box_clearance(self):
        center = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
        size = torch.tensor([
            [[0.8, 0.6], [0.6, 0.6]],
            [[1.0, 0.5], [0.5, 0.5]],
        ])
        margin = 0.25
        goal = converging_goal_xy(
            center, size, margin, torch.tensor([0.0, 1.2]),
        )
        expected = 0.5 * size.norm(dim=-1).sum(dim=-1) + margin
        self.assertTrue(torch.allclose(goal.mean(dim=1), center))
        self.assertTrue(torch.allclose(
            (goal[:, 0] - goal[:, 1]).norm(dim=-1), expected,
        ))

    def test_invalid_margin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            converging_goal_xy(
                torch.zeros(1, 2), torch.ones(1, 2, 2), -0.1,
                torch.zeros(1),
            )


if __name__ == "__main__":
    unittest.main()
