import unittest

import torch

from carry_planner.regularization import carry_path_regularization
from carry_planner.tests.test_analytic_loss import crossing_path, crossing_state
from stack_planner.history import StackPlannerObservation


def observation(state, previous_path, progress=0.0):
    batch = state.batch_size
    return StackPlannerObservation(
        state=state,
        history_tokens=torch.zeros(batch, 1, 6, 12),
        history_valid=torch.ones(batch, 1, dtype=torch.bool),
        previous_path_world=previous_path,
        previous_path_valid=torch.ones(batch, dtype=torch.bool),
        base_path_world=previous_path.clone(),
        base_path_valid=torch.ones(batch, dtype=torch.bool),
        path_progress=torch.full((batch, 2), progress),
    )


class CarryPathRegularizationTest(unittest.TestCase):
    def test_identical_direct_path_is_free(self):
        state = crossing_state()
        path = crossing_path(state)
        result = carry_path_regularization(
            {"path_world": path}, observation(state, path[:, 0]),
            torch.zeros(1),
        )
        self.assertEqual(float(result["consistency_loss"]), 0.0)
        self.assertEqual(float(result["excess_length_loss"]), 0.0)

    def test_safe_large_detour_is_bounded_and_penalized(self):
        state = crossing_state()
        base = crossing_path(state)
        detour = base.clone()
        detour[:, :, 0, 20:29, 1] += 4.0
        detour.requires_grad_(True)
        result = carry_path_regularization(
            {"path_world": detour}, observation(state, base[:, 0]),
            torch.zeros(1),
        )
        self.assertGreater(float(result["consistency_loss"]), 0.0)
        self.assertGreater(float(result["excess_length_loss"]), 0.0)
        self.assertLessEqual(float(result["consistency_loss"]), 0.25)
        self.assertLessEqual(float(result["excess_length_loss"]), 0.25)
        (result["consistency_loss"] + result["excess_length_loss"]).backward()
        self.assertGreater(float(detour.grad.abs().sum()), 0.0)

    def test_collision_risk_releases_both_geometry_terms(self):
        state = crossing_state()
        base = crossing_path(state)
        detour = base.clone()
        detour[:, :, 0, 20:29, 1] += 4.0
        result = carry_path_regularization(
            {"path_world": detour}, observation(state, base[:, 0]),
            torch.full((1,), 100.0),
        )
        self.assertEqual(float(result["consistency_loss"]), 0.0)
        self.assertEqual(float(result["excess_length_loss"]), 0.0)

    def test_executed_prefix_has_no_consistency_cost(self):
        state = crossing_state()
        base = crossing_path(state)
        changed = base.clone()
        changed[..., :9, 0] += 3.0
        result = carry_path_regularization(
            {"path_world": changed},
            observation(state, base[:, 0], progress=8.0),
            torch.zeros(1),
        )
        self.assertEqual(float(result["consistency_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
