import unittest

import torch

from carry_planner.reward import apply_invalid_plan_penalty


class CarryPlannerRewardTest(unittest.TestCase):
    def test_only_rejected_proposal_is_charged(self):
        reward, penalty = apply_invalid_plan_penalty(
            torch.tensor([1.0, 1.0, -0.5]),
            torch.tensor([True, False, False]),
            0.25,
        )
        self.assertTrue(torch.equal(
            penalty, torch.tensor([0.0, 0.25, 0.25]),
        ))
        self.assertTrue(torch.equal(
            reward, torch.tensor([1.0, 0.75, -0.75]),
        ))

    def test_curve_violation_scales_rejected_penalty_up_to_twice_base(self):
        reward, penalty = apply_invalid_plan_penalty(
            torch.ones(4),
            torch.tensor([True, False, False, False]),
            0.25,
            max_turn_deg=torch.tensor([180.0, 46.0, 69.0, 120.0]),
        )
        self.assertTrue(torch.allclose(
            penalty, torch.tensor([0.0, 0.25, 0.375, 0.5]),
        ))
        self.assertTrue(torch.allclose(reward, 1.0 - penalty))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            apply_invalid_plan_penalty(
                torch.ones(2), torch.ones(2), 0.25,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_invalid_plan_penalty(
                torch.ones(2), torch.ones(2, dtype=torch.bool), -0.1,
            )
        with self.assertRaisesRegex(ValueError, "max_turn_deg"):
            apply_invalid_plan_penalty(
                torch.ones(2), torch.ones(2, dtype=torch.bool), 0.25,
                max_turn_deg=torch.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
