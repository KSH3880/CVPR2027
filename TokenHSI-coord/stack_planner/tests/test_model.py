from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from coordinator.schema import MAX_ACCEL, MAX_SPEED, MIN_SPEED, PATH_POINTS
from coordinator.tests.common import make_state
from stack_planner.checkpoint import load_stack_checkpoint, save_stack_checkpoint
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner
from stack_planner.policy import StackPlannerActorCritic
from stack_planner.schema import STACK_SCHEMA_VERSION


class StackTrajectoryPlannerTest(unittest.TestCase):
    def test_transformer_shapes_anchors_and_mapping_input(self):
        state = make_state(batch=2)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=3))
        output = model(state.as_dict())

        self.assertTrue(any(isinstance(module, nn.TransformerEncoder) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.TransformerDecoder) for module in model.modules()))
        self.assertEqual(output["path_world"].shape, (2, 3, 2, PATH_POINTS, 2))
        self.assertEqual(output["speed"].shape, (2, 3, 2, PATH_POINTS))
        self.assertEqual(output["trajectory"].shape, (2, 3, 2, PATH_POINTS, 3))
        self.assertEqual(output["pickup_dwell"].shape, (2, 3, 2))
        self.assertEqual(output["candidate_value"].shape, (2, 3))
        self.assertEqual(output["risk_logits"].shape, (2, 3, 3))
        self.assertEqual(output["retreat_path_world"].shape, (2, 3, 2, PATH_POINTS, 2))
        self.assertEqual(output["retreat_speed"].shape, (2, 3, 2, PATH_POINTS))
        self.assertTrue(torch.allclose(output["path_world"][..., 0, :], state.root_xy[:, None], atol=1e-5))
        self.assertTrue(torch.allclose(output["path_world"][..., 16, :], state.box_xyz[:, None, :, :2], atol=1e-5))
        self.assertTrue(torch.allclose(output["path_world"][..., 32, :], state.goal_xy[:, None], atol=1e-5))
        self.assertGreaterEqual(float(output["speed"].min()), MIN_SPEED - 1e-6)
        self.assertLessEqual(float(output["speed"].max()), MAX_SPEED + 1e-6)
        self.assertTrue(torch.allclose(
            output["retreat_path_world"][..., 0, :],
            state.root_xy[:, None], atol=1e-5,
        ))

    def test_policy_samples_and_evaluates_all_standard_and_retreat_heads(self):
        state = make_state(batch=3)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        )
        output, action, log_prob, value = policy.act(state)
        evaluated_log_prob, entropy, evaluated_value, decoded = policy.evaluate(
            state, action
        )
        self.assertEqual(action.shape, (3, policy.action_dim))
        self.assertTrue(torch.allclose(log_prob, evaluated_log_prob))
        self.assertTrue(torch.allclose(value, evaluated_value))
        self.assertEqual(entropy.shape, (3,))
        self.assertEqual(decoded["retreat_path_world"].shape, (3, 1, 2, PATH_POINTS, 2))
        self.assertTrue(torch.isfinite(output["trajectory"]).all())

    def test_speed_is_acceleration_bounded_and_backward_reaches_transformer(self):
        state = make_state(batch=1)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=2))
        output = model(state)
        ds = (output["path_world"][..., 1:, :] - output["path_world"][..., :-1, :]).norm(dim=-1)
        delta_v2 = (output["speed"][..., 1:].square() - output["speed"][..., :-1].square()).abs()
        self.assertTrue((delta_v2 <= 2.0 * MAX_ACCEL * ds + 1e-5).all())

        loss = output["trajectory"].square().mean() + output["candidate_value"].mean()
        loss.backward()
        grad = model.scene_encoder.input_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_checkpoint_is_separate_and_bit_exact(self):
        torch.manual_seed(11)
        state = make_state(batch=1)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=2)).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack_v1.pth"
            save_stack_checkpoint(path, model, step=7)
            loaded, payload = load_stack_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["schema_version"], STACK_SCHEMA_VERSION)
        self.assertEqual(payload["step"], 7)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_early_v1_checkpoint_derives_redundant_retreat_contract(self):
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "early_stack_v1.pth"
            save_stack_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["retreat_path_points"]
            del payload["retreat_distance"]
            torch.save(payload, path)
            loaded, migrated = load_stack_checkpoint(path)
        self.assertEqual(migrated["retreat_path_points"], PATH_POINTS)
        self.assertEqual(migrated["retreat_distance"], model.config.retreat_distance)
        self.assertEqual(loaded.config, model.config)


if __name__ == "__main__":
    unittest.main()
