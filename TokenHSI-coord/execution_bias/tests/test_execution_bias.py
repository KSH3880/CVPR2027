from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from execution_bias.checkpoint import load_bias_checkpoint, save_bias_checkpoint
from execution_bias.data import ExecutionBiasDataset, align_actual_to_plan, nominal_times
from execution_bias.loss import execution_bias_loss
from execution_bias.model import ExecutionBiasConfig, ExecutionBiasMLP
from execution_bias.planner import attach_execution_prediction


def make_plan(batch=3, agents=2, points=33):
    t = torch.linspace(0.0, 1.0, points)
    plan = torch.zeros(batch, agents, points, 3)
    plan[:, 0, :, 0] = 4.0 * t
    plan[:, 0, :, 1] = torch.sin(torch.pi * t)
    if agents > 1:
        plan[:, 1, :, 0] = 1.0 + 3.0 * t
        plan[:, 1, :, 1] = -1.0 + t.square()
    plan[..., 2] = 1.0
    return plan


class ExecutionBiasModelTest(unittest.TestCase):
    def test_contract_zero_initialization_and_planner_gradient(self):
        plan = make_plan().requires_grad_(True)
        model = ExecutionBiasMLP()
        error = model(plan)
        self.assertEqual(error.shape, (3, 2, 33, 2))
        self.assertTrue(torch.equal(error, torch.zeros_like(error)))
        executed = model.executed_path(plan)
        self.assertTrue(torch.equal(executed, plan[..., :2]))
        executed.square().mean().backward()
        self.assertIsNotNone(plan.grad)
        self.assertGreater(float(plan.grad.abs().sum()), 0.0)

    def test_translation_rotation_equivariance(self):
        torch.manual_seed(5)
        model = ExecutionBiasMLP(ExecutionBiasConfig(hidden=64, residual_blocks=2))
        torch.nn.init.normal_(model.error_head.weight, std=0.01)
        plan = make_plan(batch=2)
        before = model(plan)
        angle = torch.tensor(0.73)
        rotation = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ])
        transformed = plan.clone()
        transformed[..., :2] = torch.einsum(
            "...i,ij->...j", plan[..., :2], rotation.T
        ) + torch.tensor([7.0, -3.0])
        after = model(transformed)
        expected = torch.einsum("...i,ij->...j", before, rotation.T)
        self.assertTrue(torch.allclose(after, expected, atol=2e-5, rtol=2e-5))

    def test_curve_weighted_loss_learns_exact_xy_target(self):
        plan = make_plan(batch=1)
        target = torch.zeros(1, 2, 33, 2)
        target[..., 10:20, 1] = -0.2
        predicted = torch.zeros_like(target, requires_grad=True)
        losses = execution_bias_loss(predicted, target, plan)
        losses["total"].backward()
        self.assertGreater(float(losses["total"]), 0.0)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_nominal_time_alignment(self):
        plan = np.zeros((1, 3, 3), dtype=np.float32)
        plan[0, :, 0] = [0.0, 1.0, 2.0]
        plan[0, :, 2] = 1.0
        self.assertTrue(np.allclose(nominal_times(plan), [[0.0, 1.0, 2.0]]))
        actual_time = np.array([0.0, 1.0, 2.0])
        actual_xy = np.zeros((3, 1, 2))
        actual_xy[:, 0, 0] = [0.0, 0.8, 1.6]
        error, valid = align_actual_to_plan(plan, actual_time, actual_xy)
        self.assertTrue(np.allclose(error[0, :, 0], [0.0, -0.2, -0.4]))
        self.assertTrue(valid.all())

    def test_dataset_and_checkpoint_round_trip(self):
        model = ExecutionBiasMLP(ExecutionBiasConfig(agents=1, hidden=32))
        plan = make_plan(batch=2, agents=1).numpy()
        error = np.zeros((2, 1, 33, 2), dtype=np.float32)
        valid = np.ones((2, 1, 33), dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "data.npz"
            checkpoint_path = Path(directory) / "bias.pth"
            np.savez(dataset_path, plan_xyv=plan, error_xy=error, valid=valid)
            dataset = ExecutionBiasDataset(dataset_path)
            save_bias_checkpoint(checkpoint_path, model, step=4)
            loaded, payload = load_bias_checkpoint(checkpoint_path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(payload["step"], 4)
        self.assertEqual(loaded.config, model.config)
        for left, right in zip(model.state_dict().values(), loaded.state_dict().values()):
            self.assertTrue(torch.equal(left, right))

    def test_joint_planner_adapter_preserves_candidates_and_gradient(self):
        model = ExecutionBiasMLP(ExecutionBiasConfig(hidden=32))
        path = make_plan(batch=6)[..., :2].reshape(2, 3, 2, 33, 2)
        path.requires_grad_(True)
        speed = torch.ones(2, 3, 2, 33, requires_grad=True)
        output = attach_execution_prediction(
            {"path_world": path, "speed": speed}, model
        )
        self.assertEqual(output["execution_error_xy"].shape, path.shape)
        self.assertTrue(torch.equal(output["executed_path_world"], path))
        output["executed_path_world"].square().mean().backward()
        self.assertGreater(float(path.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
