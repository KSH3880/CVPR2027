from __future__ import annotations

import unittest

import torch

from coordinator.geometry import resample_path_and_speed, shared_frame, shared_to_world, world_to_shared
from coordinator.losses import compute_auxiliary_loss
from coordinator.model import CoordinatorConfig, JointCoordinator
from coordinator.policy import ACTION_DIM, CoordinatorActorCritic
from coordinator.schema import CANDIDATES, MAX_ACCEL, MAX_SPEED, MIN_SPEED, PATH_POINTS, PATH_VERTICES
from coordinator.tests.common import make_state


class ModelTest(unittest.TestCase):
    def test_frame_round_trip(self):
        state = make_state()
        center, angle = shared_frame(state)
        local = world_to_shared(state.goal_xy, center, angle)
        actual = shared_to_world(local, center, angle)
        self.assertTrue(torch.allclose(actual, state.goal_xy, atol=1e-6))

    def test_shapes_anchors_and_speed_dynamics(self):
        state = make_state()
        model = JointCoordinator()
        output = model(state)
        path = output["path_world"]
        speed = output["speed"]
        self.assertEqual(path.shape, (2, CANDIDATES, 2, PATH_POINTS, 2))
        self.assertEqual(speed.shape, (2, CANDIDATES, 2, PATH_POINTS))
        self.assertTrue(torch.allclose(path[..., 0, :], state.root_xy[:, None], atol=1e-5))
        self.assertTrue(torch.allclose(path[..., 16, :], state.box_xyz[:, None, :, :2], atol=1e-5))
        self.assertTrue(torch.allclose(path[..., 32, :], state.goal_xy[:, None], atol=1e-5))
        self.assertGreaterEqual(float(speed.min()), MIN_SPEED - 1e-6)
        self.assertLessEqual(float(speed.max()), MAX_SPEED + 1e-6)
        ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
        delta_v2 = (speed[..., 1:].square() - speed[..., :-1].square()).abs()
        self.assertTrue((delta_v2 <= 2.0 * MAX_ACCEL * ds + 1e-5).all())

    def test_backward_and_dense_bridge(self):
        state = make_state()
        model = JointCoordinator()
        output = model(state)
        loss = compute_auxiliary_loss(output, state)["total"]
        loss.backward()
        grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad, 0.0)
        dense, dense_speed, end = resample_path_and_speed(
            output["path_world"][:, 0], output["speed"][:, 0]
        )
        self.assertEqual(dense.shape, (2, 2, PATH_VERTICES, 2))
        self.assertEqual(dense_speed.shape, (2, 2, PATH_VERTICES))
        self.assertEqual(end.shape, (2, 2))

    def test_stochastic_policy_contract(self):
        state = make_state()
        policy = CoordinatorActorCritic()
        output, action, log_prob, value = policy.act(state)
        self.assertEqual(action.shape, (2, ACTION_DIM))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(output["path_world"].shape[:2], (2, CANDIDATES))

    def test_analytic_prior_starts_from_straight_max_speed_plan(self):
        state = make_state()
        model = JointCoordinator(
            CoordinatorConfig(residual_scale=1.0, analytic_prior=True)
        ).eval()
        with torch.inference_mode():
            output = model(state)
        t = torch.linspace(0.0, 1.0, 17)
        first = state.root_xy[:, :, None] + t[None, None, :, None] * (
            state.box_xyz[..., :2] - state.root_xy
        )[:, :, None]
        box_xy = state.box_xyz[..., :2]
        second = box_xy[:, :, None] + t[None, None, :, None] * (
            state.goal_xy - box_xy
        )[:, :, None]
        expected = torch.cat((first, second[:, :, 1:]), dim=2)
        self.assertTrue(torch.allclose(output["path_world"], expected[:, None], atol=1e-6))
        self.assertTrue(torch.equal(output["speed"], torch.full_like(output["speed"], MAX_SPEED)))


if __name__ == "__main__":
    unittest.main()
