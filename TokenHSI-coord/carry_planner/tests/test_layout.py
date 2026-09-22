from __future__ import annotations

import unittest

import torch

from carry_planner.layout import converging_goal_xy


class ConvergingGoalLayoutTest(unittest.TestCase):
    def test_goals_cross_then_preserve_box_clearance(self):
        crossing = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
        box = torch.tensor([
            [[-2.0, -2.0], [1.0, -5.0]],
            [[-6.0, 4.0], [-3.0, 1.0]],
        ])
        size = torch.tensor([
            [[0.8, 0.6], [0.6, 0.6]],
            [[1.0, 0.5], [0.5, 0.5]],
        ])
        margin = 0.25
        goal, feasible = converging_goal_xy(
            box, crossing, size, margin,
        )
        expected = 0.5 * size.norm(dim=-1).sum(dim=-1) + margin
        self.assertTrue(feasible.all())
        self.assertTrue(torch.allclose(
            (goal[:, 0] - goal[:, 1]).norm(dim=-1), expected,
        ))
        # Goals lie beyond the shared crossing along both incoming box rays.
        incoming = crossing[:, None] - box
        outgoing = goal - crossing[:, None]
        cross2d = incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]
        self.assertTrue(torch.allclose(cross2d, torch.zeros_like(cross2d)))
        self.assertTrue(((incoming * outgoing).sum(dim=-1) > 0.0).all())

    def test_nearly_parallel_box_rays_are_rejected(self):
        goal, feasible = converging_goal_xy(
            torch.tensor([[[-2.0, 0.0], [-3.0, 0.01]]]),
            torch.zeros(1, 2), torch.ones(1, 2, 2), 0.25,
        )
        self.assertEqual(goal.shape, (1, 2, 2))
        self.assertFalse(bool(feasible[0]))

    def test_invalid_margin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            converging_goal_xy(
                torch.zeros(1, 2, 2), torch.zeros(1, 2),
                torch.ones(1, 2, 2), -0.1,
            )


if __name__ == "__main__":
    unittest.main()
