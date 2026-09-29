import unittest

import torch

from carry_planner.reward import (
    apply_invalid_plan_penalty, carry_remaining_distance,
)
from coordinator.tests.common import make_state


class CarryPlannerRewardTest(unittest.TestCase):
    def test_progress_is_continuous_across_pickup(self):
        state = make_state(batch=1)
        state.root_xy[:] = torch.tensor([[[-0.1, 0.0], [0.0, -0.1]]])
        state.box_xyz[..., :2] = 0.0
        state.goal_xy[:] = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
        before = carry_remaining_distance(state)
        self.assertTrue(torch.allclose(before, torch.tensor([5.1])))
        state.root_xy[:] = state.box_xyz[..., :2]
        state.held[:] = 1.0
        state.phase[:] = 2.0
        after = carry_remaining_distance(state)
        self.assertTrue(torch.allclose(after, torch.tensor([5.0])))
        self.assertTrue(torch.allclose(before - after, torch.tensor([0.1])))

    def test_delivery_finishes_distance_and_midroute_drop_is_penalized(self):
        state = make_state(batch=1, held=True)
        state.box_xyz[..., :2] = torch.tensor([[[4.9, 0.0], [0.0, 4.9]]])
        state.goal_xy[:] = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
        before = carry_remaining_distance(state)
        self.assertTrue(torch.allclose(before, torch.tensor([0.1]), atol=1e-6))
        state.box_xyz[..., :2] = state.goal_xy
        state.held[:] = 0.0
        state.phase[:] = 3.0
        after = carry_remaining_distance(state)
        self.assertTrue(torch.equal(after, torch.zeros_like(after)))

        state.box_xyz[..., :2] = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        state.root_xy[:] = torch.tensor([[[2.3, 0.0], [0.0, 2.3]]])
        state.held[:] = 1.0
        state.phase[:] = 2.0
        carried = carry_remaining_distance(state)
        state.held[:] = 0.0
        state.phase[:] = 0.0
        dropped = carry_remaining_distance(state)
        self.assertTrue(torch.allclose(dropped - carried, torch.tensor([0.3])))

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

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            apply_invalid_plan_penalty(
                torch.ones(2), torch.ones(2), 0.25,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_invalid_plan_penalty(
                torch.ones(2), torch.ones(2, dtype=torch.bool), -0.1,
            )


if __name__ == "__main__":
    unittest.main()
