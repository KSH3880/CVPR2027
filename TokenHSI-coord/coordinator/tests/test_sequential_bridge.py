"""CPU regressions for the sequential-stack execution bridge."""

import unittest

import torch

from coordinator.sequential_bridge import carry_rows, plan_validity, resample_plan
from coordinator.tests.common import make_state


def straight(state):
    t = torch.linspace(0, 1, 17)[None, None, :, None]
    root, box, goal = state.root_xy, state.box_xyz[..., :2], state.goal_xy
    approach = root[:, :, None] + t * (box - root)[:, :, None]
    carry = box[:, :, None] + t * (goal - box)[:, :, None]
    path = torch.cat((approach, carry[:, :, 1:]), dim=-2)[:, None]
    return path, torch.full(path.shape[:-1], 1.5)


class SequentialBridgeTest(unittest.TestCase):
    def test_phase_ownership_includes_a1_retreat(self):
        mask = carry_rows(torch.arange(6), torch.zeros(6, dtype=torch.bool))
        self.assertEqual(mask.tolist(), [
            [True, False], [True, False], [True, False],
            [False, True], [False, True], [False, False],
        ])

    def test_rehearsal_activates_both_agents(self):
        self.assertTrue(carry_rows(torch.arange(6), torch.ones(6, dtype=torch.bool)).all())

    def test_virtual_box_equal_goal_is_a_valid_degenerate_leg(self):
        state = make_state(1)
        state.goal_xy[:, 0] = state.box_xyz[:, 0, :2]
        path, speed = straight(state)
        active = torch.tensor([[True, False]])
        self.assertTrue(plan_validity(path, speed, state, active).item())
        dense, dense_speed, end = resample_plan(path[:, 0], speed[:, 0])
        self.assertTrue(torch.isfinite(dense).all())
        self.assertTrue(torch.isfinite(dense_speed).all())
        self.assertGreater(float(end[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
