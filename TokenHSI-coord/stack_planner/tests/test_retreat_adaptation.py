import unittest

import torch

from stack_planner.constraints import held_box_body_cost, retreat_endpoint_change_cost
from stack_planner.consistency import stack_trajectory_consistency_loss, build_stack_consistency_target
from test_model import make_state
from stack_planner.model import StackTrajectoryPlanner, StackPlannerConfig


class RetreatAdaptationTest(unittest.TestCase):
    def test_endpoint_soft_tolerance_mask_and_cap(self):
        current = torch.tensor([[0.05, 0.], [0.5, 0.], [100., 0.], [1., 0.]])
        result = retreat_endpoint_change_cost(current, torch.zeros_like(current),
                                              torch.tensor([True, True, True, False]))
        self.assertTrue(torch.allclose(result, torch.tensor([0., 0.16, 4., 0.])))

    def test_held_box_body_cost_respects_3d_and_held_state(self):
        bodies = torch.tensor([[[0., 0., 1.]], [[3., 0., 1.]],
                               [[0., 0., 4.]], [[0., 0., 1.]]])
        boxes = torch.tensor([[0., 0., 1.]]).repeat(4, 1)
        cost = held_box_body_cost(bodies, boxes, torch.zeros(4), torch.ones(4, 3),
                                  torch.tensor([1., 1., 1., 0.]))
        self.assertGreater(float(cost[0]), 0)
        self.assertTrue(torch.equal(cost[1:], torch.zeros(3)))

    def test_path_consistency_scale_reduces_loss_not_mask_denominator(self):
        state = make_state(batch=1)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        output = model(state)
        target = build_stack_consistency_target(output, state, state, 0.,
                                                 torch.ones(1, dtype=torch.bool))
        target['position'] = target['position'] + 0.2
        original = stack_trajectory_consistency_loss(output, state, target)['total']
        target['position_scale'] = torch.full_like(target['valid'], 0.1, dtype=torch.float32)
        reduced = stack_trajectory_consistency_loss(output, state, target)['total']
        self.assertTrue(torch.allclose(reduced, original * 0.1))
