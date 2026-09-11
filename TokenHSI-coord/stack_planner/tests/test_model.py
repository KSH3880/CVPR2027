from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from coordinator.schema import AGENTS, MIN_SPEED, PATH_POINTS
from coordinator.tests.common import make_state
from stack_planner.checkpoint import load_stack_checkpoint, save_stack_checkpoint
from stack_planner.consistency import (
    build_stack_consistency_target, stack_trajectory_consistency_loss,
)
from stack_planner.constraints import (
    free_path_validity, ordered_box_goal_visit, project_points_to_segments,
)
from stack_planner.execution import execution_view
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner
from stack_planner.policy import StackPlannerActorCritic
from stack_planner.schema import STACK_PATH_POINTS, STACK_SCHEMA_VERSION


class StackTrajectoryPlannerTest(unittest.TestCase):
    def test_time_shifted_consistency_is_zero_for_same_constant_plan(self):
        state = make_state(batch=2)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with torch.no_grad():
            output = model(state)
            target = build_stack_consistency_target(
                output, state, state, 0.0,
                torch.ones(2, dtype=torch.bool),
            )
        loss = stack_trajectory_consistency_loss(output, state, target)
        self.assertLess(float(loss["total"]), 1e-7)
        self.assertGreater(float(loss["valid_fraction"]), 0.0)

    def test_consistency_masks_reset_and_phase_change(self):
        state = make_state(batch=2)
        changed = state.clone()
        changed.phase[1, 0] = 3.0
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with torch.no_grad():
            output = model(state)
            target = build_stack_consistency_target(
                output, state, changed, 0.2,
                torch.tensor([False, True]),
            )
        self.assertFalse(target["valid"][0].any())
        self.assertFalse(target["valid"][1, :, 0].any())
        self.assertTrue(target["valid"][1, :, 1].any())

    def test_consistency_reaches_unified_path_head(self):
        state = make_state(batch=2)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        with torch.no_grad():
            model.heads.path[-1].bias[2] = 1.0
        output = model(state)
        with torch.no_grad():
            target = build_stack_consistency_target(
                output, state, state, 0.0,
                torch.ones(2, dtype=torch.bool),
            )
            target["position"].add_(0.2)
        loss = stack_trajectory_consistency_loss(output, state, target)
        self.assertGreater(float(loss["total"]), 0.0)
        loss["total"].backward()
        path_grad = model.heads.path[-1].weight.grad
        self.assertIsNotNone(path_grad)
        self.assertGreater(float(path_grad.abs().sum()), 0.0)

    def test_transformer_shapes_root_anchor_and_mapping_input(self):
        state = make_state(batch=2)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=3))
        output = model(state.as_dict())

        self.assertTrue(any(isinstance(module, nn.TransformerEncoder) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.TransformerDecoder) for module in model.modules()))
        self.assertEqual(output["path_world"].shape, (2, 3, 2, STACK_PATH_POINTS, 2))
        self.assertNotIn("speed", output)
        self.assertNotIn("acceleration", output)
        self.assertNotIn("trajectory", output)
        self.assertEqual(set(output), {"path_local", "path_world"})
        self.assertTrue(torch.allclose(output["path_world"][..., 0, :], state.root_xy[:, None], atol=1e-5))
        # Zero initialization retains the executable root->box->goal prior,
        # but only root remains structurally anchored.
        self.assertTrue(torch.allclose(output["path_world"][..., 10, :], state.box_xyz[:, None, :, :2], atol=1e-5))
        self.assertTrue(torch.allclose(output["path_world"][..., 21, :], state.goal_xy[:, None], atol=1e-5))
        self.assertTrue(torch.allclose(
            output["path_world"][:, :, 0, -1], state.goal_xy[:, None, 0], atol=1e-5,
        ))

        with torch.no_grad():
            # First agent's free junction and endpoint deltas.
            model.heads.path[-1].bias[0] = 0.5
            model.heads.path[-1].bias[2] = 0.5
            moved = model(state)
        self.assertFalse(torch.allclose(
            moved["path_world"][..., 10, :], state.box_xyz[:, None, :, :2]
        ))
        self.assertFalse(torch.allclose(
            moved["path_world"][..., 21, :], state.goal_xy[:, None]
        ))
        self.assertTrue(torch.allclose(
            moved["path_world"][..., 0, :], state.root_xy[:, None], atol=1e-5
        ))

    def test_physical_completion_does_not_hard_switch_the_path(self):
        state = make_state(batch=1)
        state.phase[:, 0] = 3.0
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with torch.no_grad():
            model.heads.path[-1].bias[AGENTS * 6 * 2] = 0.5
            output = model(state)
        self.assertFalse(torch.allclose(
            output["path_world"][:, :, 0, -1], state.goal_xy[:, None, 0]
        ))
        self.assertNotIn("route_required", output)
        self.assertNotIn("retreat_path_world", output)
        self.assertNotIn("agent1_full_path_world", output)

    def test_segment_projection_and_ordered_visit_penalty(self):
        path = torch.tensor([[[[
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        ]]]])
        box = torch.tensor([[[[0.5, 0.2]]]])
        goal = torch.tensor([[[[2.5, -0.1]]]])
        projection = project_points_to_segments(path, box)
        self.assertAlmostEqual(float(projection["distance2"].min()), 0.04, places=6)
        ordered = ordered_box_goal_visit(path, box, goal, tolerance=0.0)
        self.assertAlmostEqual(float(ordered["box_distance"]), 0.2, places=6)
        self.assertAlmostEqual(float(ordered["goal_distance"]), 0.1, places=6)
        self.assertLessEqual(int(ordered["box_segment"]), int(ordered["goal_segment"]))

        reverse = ordered_box_goal_visit(path, goal, box, tolerance=0.0)
        self.assertGreater(float(reverse["penalty"]), float(ordered["penalty"]))

    def test_execution_interpolates_to_goal_then_translates_same_suffix(self):
        points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0],
        ])
        path = points[None, None].repeat(1, 2, 1, 1)
        box = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        goal = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])

        carry = execution_view(
            path, box, goal, torch.zeros(1, 2, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(carry[..., 0, :], path[..., 0, :]))
        self.assertTrue(torch.equal(carry[..., -1, :], goal))

        retreat_mask = torch.tensor([[True, False]])
        mixed = execution_view(path, box, goal, retreat_mask)
        self.assertTrue(torch.equal(mixed[..., 0, :], path[..., 0, :]))
        self.assertTrue(torch.allclose(mixed[0, 0, -1], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(mixed[0, 1, -1], goal[0, 1]))

    def test_visit_penalty_is_differentiable_and_free_validity_needs_only_root(self):
        state = make_state(batch=1)
        path = torch.stack((
            torch.linspace(0, 1, PATH_POINTS),
            torch.zeros(PATH_POINTS),
        ), dim=-1).reshape(1, 1, 1, PATH_POINTS, 2).repeat(1, 1, 2, 1, 1)
        path = path.clone().requires_grad_(True)
        box = torch.tensor([[[[0.25, 0.3], [0.25, 0.3]]]])
        goal = torch.tensor([[[[0.75, -0.2], [0.75, -0.2]]]])
        visit = ordered_box_goal_visit(path, box, goal, tolerance=0.05)
        visit["penalty"].mean().backward()
        self.assertGreater(float(path.grad.abs().sum()), 0.0)

        world_path = state.root_xy[:, None, :, None, :].expand(
            -1, 1, -1, STACK_PATH_POINTS, -1
        ).clone()
        # Deliberately do not pass through box or goal; root-only validity
        # still accepts this finite, bounded compatibility path.
        speed = torch.full(world_path.shape[:-1], MIN_SPEED)
        valid = free_path_validity(
            world_path, speed, state.root_xy,
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertTrue(bool(valid.item()))

    def test_policy_samples_and_evaluates_unified_head(self):
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
        self.assertEqual(decoded["path_world"].shape, (3, 1, 2, STACK_PATH_POINTS, 2))
        self.assertTrue(torch.isfinite(output["path_world"]).all())

    def test_path_backward_reaches_transformer(self):
        state = make_state(batch=1)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=2))
        )
        _, value, _ = policy.distribution(state)
        loss = value.mean()
        loss.backward()
        grad = policy.planner.scene_encoder.input_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_checkpoint_is_separate_and_bit_exact(self):
        torch.manual_seed(11)
        state = make_state(batch=1)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=2)).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack_v5.pth"
            save_stack_checkpoint(path, model, step=7)
            loaded, payload = load_stack_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["schema_version"], STACK_SCHEMA_VERSION)
        self.assertEqual(payload["step"], 7)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_v4_schema_is_rejected_after_path_only_action_change(self):
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_schema.pth"
            save_stack_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["schema_version"] = "tokenhsi-stack-planner-v4"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                load_stack_checkpoint(path)

    def test_checkpoint_records_single_path_contract(self):
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack_v5.pth"
            save_stack_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            loaded, migrated = load_stack_checkpoint(path)
        self.assertEqual(migrated["path_points"], PATH_POINTS)
        self.assertNotIn("retreat_path_points", migrated)
        self.assertNotIn("endpoint_distance", migrated)
        self.assertTrue(migrated["path_only"])
        self.assertEqual(loaded.config, model.config)


if __name__ == "__main__":
    unittest.main()
