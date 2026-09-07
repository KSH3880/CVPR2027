from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from coordinator.schema import MAX_SPEED
from coordinator.simple_checkpoint import load_simple_checkpoint, save_simple_checkpoint
from coordinator.simple_model import (
    SIMPLE_ACTION_DIM,
    SimpleCoordinatorConfig,
    SimpleJointCoordinator,
)
from coordinator.simple_policy import SimpleCoordinatorActorCritic
from coordinator.tests.common import make_state


class SimpleCoordinatorTest(unittest.TestCase):
    def test_zero_init_is_exact_analytic_plan(self):
        state = make_state()
        model = SimpleJointCoordinator().eval()
        with torch.inference_mode():
            output = model(state)
        t = torch.linspace(0.0, 1.0, 17)
        box = state.box_xyz[..., :2]
        first = state.root_xy[:, :, None] + t[None, None, :, None] * (
            box - state.root_xy
        )[:, :, None]
        second = box[:, :, None] + t[None, None, :, None] * (
            state.goal_xy - box
        )[:, :, None]
        expected = torch.cat((first, second[:, :, 1:]), dim=2)
        self.assertTrue(torch.allclose(output["path_world"], expected[:, None], atol=1e-6))
        self.assertTrue(torch.equal(output["speed"], torch.full_like(output["speed"], MAX_SPEED)))

    def test_policy_has_only_four_actions_and_preserves_pickup_leg(self):
        state = make_state()
        policy = SimpleCoordinatorActorCritic()
        output, action, log_prob, value = policy.act(state)
        self.assertEqual(action.shape, (state.batch_size, SIMPLE_ACTION_DIM))
        self.assertEqual(log_prob.shape, (state.batch_size,))
        self.assertEqual(value.shape, (state.batch_size,))
        box = state.box_xyz[..., :2]
        t = torch.linspace(0.0, 1.0, 17)
        expected = state.root_xy[:, :, None] + t[None, None, :, None] * (
            box - state.root_xy
        )[:, :, None]
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, :17], expected, atol=1e-6))

    def test_checkpoint_round_trip(self):
        torch.manual_seed(3)
        state = make_state(1)
        model = SimpleJointCoordinator().eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b0.pth"
            save_simple_checkpoint(path, model, step=7)
            loaded, payload = load_simple_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
            self.assertEqual(payload["step"], 7)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_safe_bow_scales_lateral_by_short_carry_leg(self):
        state = make_state(batch=2)
        state.goal_xy[:] = state.box_xyz[..., :2] + torch.tensor([0.2, 0.0])
        model = SimpleJointCoordinator(SimpleCoordinatorConfig(safe_bow=True))
        action = torch.zeros(2, SIMPLE_ACTION_DIM)
        action[:, :2] = 10.0
        output = model.decode(state, action, torch.zeros(2))
        self.assertTrue((output["lateral"].abs() <= 0.2 + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
