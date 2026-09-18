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
    retreat_box_clearance,
)
from stack_planner.execution import execution_view, retreat_box_geometry
from stack_planner.history import StackHistoryBuffer, project_path_progress
from stack_planner.model import (
    StackPlannerConfig, StackTrajectoryPlanner, _future_point_weight,
)
from stack_planner.policy import StackPlannerActorCritic
from stack_planner.schema import (
    STACK_PATH_DELTA_DIM, STACK_PATH_POINTS, STACK_SCHEMA_VERSION,
)


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
            model.heads.paths[0][-1].bias[2] = 1.0
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
        path_grad = model.heads.paths[0][-1].weight.grad
        self.assertIsNotNone(path_grad)
        self.assertGreater(float(path_grad.abs().sum()), 0.0)

    def test_scene_token_encoder_and_independent_full_path_heads(self):
        state = make_state(batch=2)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=3))
        output = model(state.as_dict())
        _, raw = model.raw_heads(state)

        self.assertTrue(any(isinstance(module, nn.TransformerEncoder) for module in model.modules()))
        self.assertFalse(any(isinstance(module, nn.TransformerDecoder) for module in model.modules()))
        self.assertEqual(model.scene_encoder.scene_token.shape, (1, 1, 128))
        self.assertEqual(len(model.scene_encoder.tokenizers), 3)
        self.assertEqual(len(model.heads.paths), 3)
        self.assertIsNot(model.heads.paths[0], model.heads.paths[1])
        self.assertEqual(raw["path_delta_raw"].shape, (2, 3, STACK_PATH_DELTA_DIM))
        self.assertEqual(raw["reference_path_local"].shape, (2, 2, 33, 2))
        self.assertEqual(raw["candidate_logits"].shape, (2, 3))
        self.assertEqual(output["path_world"].shape, (2, 1, 2, STACK_PATH_POINTS, 2))
        self.assertNotIn("speed", output)
        self.assertNotIn("acceleration", output)
        self.assertNotIn("trajectory", output)
        self.assertEqual(set(output), {
            "path_local", "path_world", "selected_candidate", "candidate_logits",
        })
        self.assertTrue(torch.allclose(output["path_world"][..., 0, :], state.root_xy[:, None], atol=1e-5))
        self.assertLess(float((output["path_world"][..., 10, :] - state.box_xyz[:, None, :, :2]).abs().max()), 0.1)
        self.assertLess(float((output["path_world"][..., 21, :] - state.goal_xy[:, None]).abs().max()), 0.1)

    def test_physical_completion_does_not_hard_switch_the_path(self):
        state = make_state(batch=1)
        state.phase[:, 0] = 3.0
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with torch.no_grad():
            model.heads.paths[0][-1].bias[(STACK_PATH_POINTS - 2) * 2] = 0.5
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

    def test_execution_attaches_suffix_without_moving_world_endpoint(self):
        points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0],
        ])
        path = points[None, None].repeat(1, 2, 1, 1)
        box = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        goal = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])

        carry = execution_view(
            path, box, goal, torch.zeros(1, 2, dtype=torch.bool), goal,
        )
        self.assertTrue(torch.equal(carry[..., 0, :], path[..., 0, :]))
        self.assertTrue(torch.equal(carry[..., -1, :], goal))

        retreat_mask = torch.tensor([[True, False]])
        mixed = execution_view(path, box, goal, retreat_mask, goal)
        self.assertTrue(torch.equal(mixed[0, 0, 0], goal[0, 0]))
        self.assertTrue(torch.equal(mixed[0, 1, 0], path[0, 1, 0]))
        self.assertTrue(torch.equal(mixed[0, 0, -1], path[0, 0, -1]))
        self.assertTrue(torch.equal(mixed[0, 1, -1], goal[0, 1]))

    def test_zero_length_retreat_suffix_keeps_goal_not_root_as_endpoint(self):
        points = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [2., 0.]])
        path = points[None, None].repeat(1, 2, 1, 1)
        box = torch.tensor([[[1., 0.], [1., 0.]]])
        goal = torch.tensor([[[2., 0.], [2., 0.]]])
        retreat = execution_view(
            path, box, goal, torch.tensor([[True, False]]), goal,
        )
        self.assertTrue(torch.equal(retreat[0, 0, 0], goal[0, 0]))
        self.assertTrue(torch.equal(retreat[0, 0, -1], path[0, 0, -1]))
        self.assertTrue(torch.equal(retreat[0, 0, -1], retreat[0, 0, 0]))

    def test_retreat_execution_drops_already_traversed_suffix(self):
        points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        ])
        path = points[None, None].repeat(1, 2, 1, 1)
        box = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        goal = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
        root = torch.tensor([[[2.6, 0.2], [2.0, 0.0]]])
        retreat = execution_view(
            path, box, goal, torch.tensor([[True, False]]), root,
        )
        self.assertAlmostEqual(float(retreat[0, 0, 0, 0]), 2.6, places=5)
        self.assertAlmostEqual(float(retreat[0, 0, 0, 1]), 0.0, places=5)
        self.assertTrue(torch.equal(retreat[0, 0, -1], path[0, 0, -1]))

    def test_retreat_clearance_prefers_moving_away_over_crossing_box(self):
        safe = torch.tensor([[
            [0.0, -0.7], [0.0, -1.0], [0.0, -1.3], [0.0, -1.6],
        ]], requires_grad=True)
        crossing = torch.tensor([[
            [0.0, -0.7], [0.0, -0.2], [0.0, 0.2], [0.0, 0.7],
        ]], requires_grad=True)
        box_xy = torch.zeros(1, 2)
        box_yaw = torch.zeros(1)
        box_size = torch.full((1, 2), 0.4)
        safe_cost = retreat_box_clearance(
            safe, box_xy, box_yaw, box_size,
        )["penalty"]
        crossing_cost = retreat_box_clearance(
            crossing, box_xy, box_yaw, box_size,
        )["penalty"]
        self.assertLess(float(safe_cost), float(crossing_cost))
        crossing_cost.mean().backward()
        self.assertGreater(float(crossing.grad.abs().sum()), 0.0)

        ends_inside = torch.tensor([[
            [0.0, -0.7], [0.0, -1.1], [0.0, -0.9], [0.0, -0.5],
        ]])
        inside_result = retreat_box_clearance(
            ends_inside, box_xy, box_yaw, box_size,
        )
        self.assertGreater(float(inside_result["endpoint_overlap"]), 0.0)
        self.assertEqual(float(
            retreat_box_clearance(safe, box_xy, box_yaw, box_size)[
                "endpoint_overlap"
            ]
        ), 0.0)

        state = make_state(batch=1)
        state.box_xyz[0, 0, :2] = box_xy[0]
        state.box_heading[0, 0] = box_yaw[0]
        state.box_size_xy[0, 0] = box_size[0]
        joint_path = safe.detach()[:, None].repeat(1, 2, 1, 1)
        measured = retreat_box_geometry(
            joint_path, state, torch.tensor([True]),
        )
        self.assertTrue(torch.allclose(measured["penalty"], safe_cost.detach()))

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
            StackTrajectoryPlanner(StackPlannerConfig(candidates=4))
        )
        output, action, log_prob, value = policy.act(state)
        evaluated_log_prob, entropy, evaluated_value, decoded = policy.evaluate(
            state, action
        )
        self.assertEqual(action.shape, (3, policy.action_dim))
        self.assertEqual(policy.action_dim, STACK_PATH_DELTA_DIM + 1)
        self.assertTrue(((action[:, 0] >= 0) & (action[:, 0] < 4)).all())
        self.assertTrue(torch.allclose(log_prob, evaluated_log_prob))
        self.assertTrue(torch.allclose(value, evaluated_value))
        self.assertEqual(entropy.shape, (3,))
        self.assertEqual(decoded["path_world"].shape, (3, 1, 2, STACK_PATH_POINTS, 2))
        self.assertTrue(torch.isfinite(output["path_world"]).all())

    def test_policy_samples_every_candidate_from_same_scene(self):
        state = make_state(batch=3)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=4))
        )
        output, action, log_prob, value = policy.sample_all(state)
        self.assertEqual(action.shape, (3, 4, policy.action_dim))
        self.assertEqual(log_prob.shape, (3, 4))
        self.assertEqual(value.shape, (3,))
        self.assertEqual(
            output["path_world"].shape,
            (3, 4, AGENTS, STACK_PATH_POINTS, 2),
        )
        expected = torch.arange(4).reshape(1, 4).expand(3, -1)
        self.assertTrue(torch.equal(action[..., 0].long(), expected))

    def test_executed_action_dimensions_do_not_change_log_probability(self):
        state = make_state(batch=1)
        previous = torch.zeros(1, AGENTS, STACK_PATH_POINTS, 2)
        previous[..., 0] = torch.arange(STACK_PATH_POINTS)
        state.root_xy[0, 0] = torch.tensor([8.0, 0.0])
        state.root_xy[0, 1] = torch.tensor([0.0, 0.0])
        buffer = StackHistoryBuffer(1, 1, state.device)
        buffer.commit_path(previous)
        observation = buffer.observe(state)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        )
        _, action, log_prob, _ = policy.act(observation, deterministic=True)
        modified = action.clone()
        # A1 point 1..8 XY corrections are behind progress=8.
        modified[:, 1:1 + 8 * 2] += 5.0
        modified_log_prob, _, _, _ = policy.evaluate(observation, modified)
        self.assertTrue(torch.allclose(log_prob, modified_log_prob))

    def test_only_selected_full_path_head_gets_continuous_action_credit(self):
        state = make_state(batch=2)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=3))
        )
        paths, _, _ = policy.distribution(state)
        action = torch.cat((
            torch.zeros(2, 1),
            paths.loc[:, 0].detach() + 0.1,
        ), dim=-1)
        log_prob, _, _, _ = policy.evaluate(state, action)
        (-log_prob.mean()).backward()
        selected_grad = policy.planner.heads.paths[0][-1].weight.grad
        self.assertIsNotNone(selected_grad)
        self.assertGreater(float(selected_grad.abs().sum()), 0.0)
        for candidate in (1, 2):
            grad = policy.planner.heads.paths[candidate][-1].weight.grad
            self.assertTrue(grad is None or float(grad.abs().sum()) == 0.0)

    def test_independent_heads_start_distinct_and_have_diversity_objective(self):
        torch.manual_seed(3)
        state = make_state(batch=2)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=4))
        )
        _, raw = policy.planner.raw_heads(state)
        self.assertGreater(float((
            raw["path_delta_raw"][:, 0] - raw["path_delta_raw"][:, 1]
        ).abs().sum()), 0.0)
        diversity = policy.diversity(state, margin=0.25)
        self.assertTrue(torch.isfinite(diversity["loss"]))
        self.assertGreaterEqual(float(diversity["distance"]), 0.0)
        self.assertTrue(torch.isfinite(diversity["smoothness_loss"]))

    def test_smoothness_objective_penalizes_zigzag_mean_correction(self):
        state = make_state(batch=1)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        )
        final = policy.planner.heads.paths[0][-1]
        nn.init.zeros_(final.weight)
        raw = torch.zeros(AGENTS, STACK_PATH_POINTS - 1, 2)
        raw[..., 0] = torch.where(
            torch.arange(STACK_PATH_POINTS - 1) % 2 == 0, 1.0, -1.0,
        )
        final.bias.data.copy_(raw.reshape(-1))
        regularity = policy.diversity(state)
        self.assertGreater(float(regularity["smoothness_loss"]), 0.0)
        regularity["smoothness_loss"].backward()
        self.assertGreater(float(final.bias.grad.abs().sum()), 0.0)

    def test_pointwise_delta_exploration_scales(self):
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=1))
        )
        std = policy.action_log_std.exp().reshape(1, AGENTS, 32, 2)
        self.assertTrue(torch.allclose(std[:, :, 9], torch.full((1, 2, 2), 0.03)))
        self.assertTrue(torch.allclose(std[:, :, 20], torch.full((1, 2, 2), 0.03)))
        self.assertTrue(torch.allclose(std[:, :, 5], torch.full((1, 2, 2), 0.12)))
        self.assertTrue(torch.allclose(std[:, :, -1], torch.full((1, 2, 2), 0.20)))

    def test_history_observation_masks_reset_and_drives_planner(self):
        state = make_state(batch=2)
        buffer = StackHistoryBuffer(2, 4, state.device)
        first = buffer.observe(state)
        self.assertEqual(first.history_valid.sum(dim=1).tolist(), [1, 1])
        moved = state.clone()
        moved.root_xy[:, 0, 0] += 0.25
        second = buffer.observe(moved)
        self.assertEqual(second.history_valid.sum(dim=1).tolist(), [2, 2])
        third = buffer.observe(moved, reset_mask=torch.tensor([True, False]))
        self.assertEqual(third.history_valid.sum(dim=1).tolist(), [1, 3])
        model = StackTrajectoryPlanner(StackPlannerConfig(
            candidates=1, history_steps=4,
        ))
        output = model(third)
        self.assertEqual(
            output["path_world"].shape,
            (2, 1, AGENTS, STACK_PATH_POINTS, 2),
        )

    def test_counterfactual_history_preview_does_not_mutate_buffer(self):
        state = make_state(batch=2)
        buffer = StackHistoryBuffer(2, 4, state.device)
        buffer.observe(state)
        saved_tokens = buffer.tokens.clone()
        saved_valid = buffer.valid.clone()
        preview = buffer.observe(
            state, reset_mask=torch.tensor([True, False]), commit=False,
        )
        self.assertTrue(torch.equal(buffer.tokens, saved_tokens))
        self.assertTrue(torch.equal(buffer.valid, saved_valid))
        self.assertEqual(preview.history_valid.sum(dim=1).tolist(), [1, 2])

    def test_selected_trajectory_becomes_next_correction_reference(self):
        state = make_state(batch=2)
        buffer = StackHistoryBuffer(2, 4, state.device)
        first = buffer.observe(state)
        model = StackTrajectoryPlanner(StackPlannerConfig(
            candidates=1, history_steps=4, delta_scale=0.5,
        )).eval()
        for head in model.heads.paths:
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            initial = model(first)
        committed = initial["path_world"][:, 0].clone()
        committed[..., 5:, 1] += 0.2
        buffer.commit_path(committed)
        moved = state.clone()
        direction = committed[:, :, 1] - committed[:, :, 0]
        moved.root_xy = committed[:, :, 0] + 0.6 * direction
        second = buffer.observe(moved)
        with torch.no_grad():
            refined = model(second)
        self.assertTrue(torch.allclose(
            refined["path_world"][:, 0], committed, atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            refined["path_world"][:, 0, :, 0], committed[:, :, 0], atol=1e-6,
        ))
        self.assertFalse(torch.allclose(
            refined["path_world"][:, 0, :, 0], moved.root_xy, atol=1e-6,
        ))
        buffer.reset(torch.tensor([True, False]))
        reset_observation = buffer.observe(state, commit=False)
        self.assertFalse(bool(reset_observation.previous_path_valid[0]))
        self.assertTrue(bool(reset_observation.previous_path_valid[1]))

    def test_previous_path_keeps_origin_and_masks_executed_prefix(self):
        previous = torch.zeros(1, AGENTS, STACK_PATH_POINTS, 2)
        previous[..., 0] = torch.arange(STACK_PATH_POINTS)
        root = previous[..., 0, :].clone()
        root[:, 0, 0] = 8.0
        progress = project_path_progress(previous, root)
        weight = _future_point_weight(progress, torch.tensor([True]))
        self.assertAlmostEqual(float(progress[0, 0]), 8.0)
        self.assertTrue(torch.equal(previous[0, 0, 0], torch.tensor([0., 0.])))
        self.assertTrue(torch.equal(weight[0, 0, :9], torch.zeros(9)))
        self.assertAlmostEqual(float(weight[0, 0, 9]), 0.5)
        self.assertAlmostEqual(float(weight[0, 0, 10]), 1.0)
        later_root = root.clone()
        later_root[:, 0, 0] = 6.0
        monotonic = project_path_progress(previous, later_root, progress)
        self.assertEqual(float(monotonic[0, 0]), 8.0)

    def test_path_backward_reaches_transformer(self):
        state = make_state(batch=1)
        policy = StackPlannerActorCritic(
            StackTrajectoryPlanner(StackPlannerConfig(candidates=2))
        )
        value = policy.value(state)
        loss = value.mean()
        loss.backward()
        grad = policy.planner.scene_encoder.tokenizers[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_checkpoint_is_separate_and_bit_exact(self):
        torch.manual_seed(11)
        state = make_state(batch=1)
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=2)).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack_v12.pth"
            save_stack_checkpoint(path, model, step=7)
            loaded, payload = load_stack_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["schema_version"], STACK_SCHEMA_VERSION)
        self.assertEqual(payload["step"], 7)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_v11_schema_is_rejected_after_joint_a2_preplan_change(self):
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_schema.pth"
            save_stack_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["schema_version"] = "tokenhsi-stack-planner-v11"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                load_stack_checkpoint(path)

    def test_checkpoint_records_single_path_contract(self):
        model = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack_v12.pth"
            save_stack_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            loaded, migrated = load_stack_checkpoint(path)
        self.assertEqual(migrated["path_points"], PATH_POINTS)
        self.assertNotIn("retreat_path_points", migrated)
        self.assertNotIn("endpoint_distance", migrated)
        self.assertTrue(migrated["path_only"])
        self.assertTrue(migrated["full_candidate_rollout"])
        self.assertTrue(migrated["previous_trajectory_correction"])
        self.assertTrue(migrated["fixed_origin_reference"])
        self.assertTrue(migrated["future_action_mask"])
        self.assertTrue(migrated["projected_executor_resume"])
        self.assertTrue(migrated["joint_a2_preplan"])
        self.assertTrue(migrated["deferred_a2_carry_goal"])
        self.assertTrue(migrated["smooth_path_regularization"])
        self.assertEqual(loaded.config, model.config)


if __name__ == "__main__":
    unittest.main()
