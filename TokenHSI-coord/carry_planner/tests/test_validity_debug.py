import unittest

import torch

from carry_planner.tests.test_analytic_loss import crossing_path, crossing_state
from carry_planner.validity_debug import carry_plan_validity_debug


class CarryValidityDebugTest(unittest.TestCase):
    def test_straight_path_passes_curve_check(self):
        state = crossing_state()
        path = crossing_path(state)[:, 0]
        speed = torch.ones(path.shape[:-1])
        result = carry_plan_validity_debug(state, path, speed)
        self.assertTrue(bool(result["curve"].all()))
        self.assertFalse(bool(result["zero_turn_false_reject"].any()))

    def test_duplicate_point_exposes_zero_segment_false_reject(self):
        state = crossing_state()
        path = crossing_path(state)[:, 0]
        path[:, :, 10] = path[:, :, 9]
        speed = torch.ones(path.shape[:-1])
        result = carry_plan_validity_debug(state, path, speed)
        self.assertFalse(bool(result["curve"].all()))
        self.assertTrue(bool(result["zero_safe_curve"].all()))
        self.assertTrue(bool(result["zero_turn_false_reject"].all()))
        self.assertTrue(bool(result["relevant_degenerate_turn"].all()))


if __name__ == "__main__":
    unittest.main()
