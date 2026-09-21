import unittest

import torch

from carry_planner.analytic_loss import carry_analytic_collision_loss
from coordinator.schema import CoordinatorState


def crossing_state():
    root = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]])
    box_xy = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])
    goal = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    box = torch.cat((box_xy, torch.zeros(1, 2, 1)), dim=-1)
    return CoordinatorState(
        root_xy=root,
        heading=torch.zeros(1, 2),
        root_vel_xy=torch.zeros(1, 2, 2),
        box_xyz=box,
        box_heading=torch.zeros(1, 2),
        box_vel_xy=torch.zeros(1, 2, 2),
        box_size_xy=torch.full((1, 2, 2), 0.5),
        goal_xy=goal,
        held=torch.zeros(1, 2),
        phase=torch.zeros(1, 2),
    )


def crossing_path(state):
    unit = torch.linspace(0.0, 1.0, 17)
    box = state.box_xyz[..., :2]
    approach = state.root_xy[:, :, None] + unit[None, None, :, None] * (
        box - state.root_xy
    )[:, :, None]
    carry = box[:, :, None] + unit[None, None, :, None] * (
        state.goal_xy - box
    )[:, :, None]
    return torch.cat((approach, carry[:, :, 1:]), dim=2)[:, None]


class CarryAnalyticLossTest(unittest.TestCase):
    def test_collision_gradient_reaches_path_but_not_speed(self):
        state = crossing_state()
        path = crossing_path(state).requires_grad_(True)
        speed = torch.full(path.shape[:-1], 1.36, requires_grad=True)
        result = carry_analytic_collision_loss(
            {"path_world": path, "speed": speed},
            state, torch.ones(1, dtype=torch.bool), focus_steps=8,
        )
        self.assertGreater(float(result["loss"]), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(path.grad)
        self.assertGreater(float(path.grad.abs().sum()), 0.0)
        self.assertIsNone(speed.grad)

    def test_noninitial_plan_has_exact_zero_loss(self):
        state = crossing_state()
        path = crossing_path(state).requires_grad_(True)
        speed = torch.full(path.shape[:-1], 1.36, requires_grad=True)
        result = carry_analytic_collision_loss(
            {"path_world": path, "speed": speed},
            state, torch.zeros(1, dtype=torch.bool), focus_steps=8,
        )
        self.assertEqual(float(result["loss"]), 0.0)
        result["loss"].backward()
        self.assertEqual(float(path.grad.abs().sum()), 0.0)
        self.assertIsNone(speed.grad)


if __name__ == "__main__":
    unittest.main()
