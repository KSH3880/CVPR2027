from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from coordinator.c2_checkpoint import (
    C2_ACTION_DIM,
    C2_CONFLICT_WINDOW_ACTION_DIM,
    C2_JOINT_POINT_ACTION_DIM,
    C2_SLOWDOWN_WINDOW_ACTION_DIM,
    C2_WAYPOINT_ACTION_DIM,
    c2_config,
    load_c2_checkpoint,
    save_c2_checkpoint,
)
from coordinator.c2_losses import (
    build_trajectory_consistency_target,
    compute_c2_auxiliary_loss,
    trajectory_consistency_loss,
)
from coordinator.geometry import (
    build_waypoint_paths, physical_point_speed_profile, state_to_tokens,
)
from coordinator.model import MLP_CONFLICT_FEATURES, JointCoordinator
from coordinator.planner import (
    apply_fixed_priority, candidate_costs, pointwise_proximity_risk,
)
from coordinator.policy import CoordinatorActorCritic
from coordinator.schema import (
    MAX_ACCEL, MIN_SPEED, MAX_SPEED, PATH_POINTS, WAYPOINT_RESIDUAL_POINTS,
)
from coordinator.tests.common import make_state


class C2CoordinatorTest(unittest.TestCase):
    def test_learned_priority_head_is_neutral_and_checkpointed(self):
        state = make_state(batch=3)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            physical_speed_caps=True, learned_priority=True,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        self.assertEqual(before["priority_logits"].shape, (3, 2))
        self.assertTrue(torch.equal(
            before["priority_logits"],
            torch.zeros_like(before["priority_logits"]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c11_learned_priority.pth"
            save_c2_checkpoint(path, model, step=1)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertTrue(payload["model_config"]["learned_priority"])
        self.assertEqual(payload["priority_classes"], 2)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_physical_point_caps_create_reachable_pre_and_post_profiles(self):
        x = torch.linspace(0.0, 8.0, PATH_POINTS)
        path = torch.stack((x, torch.zeros_like(x)), dim=-1)
        path = path[None, None, None].repeat(1, 1, 2, 1, 1)
        raw = torch.full(path.shape[:-1], -5.0)
        raw[..., 0, 12] = 5.0
        raw[..., 0, 24] = 3.0
        speed, acceleration, request = physical_point_speed_profile(path, raw)

        self.assertEqual(speed.shape, raw.shape)
        self.assertEqual(request.shape, raw.shape)
        self.assertEqual(int(speed[..., 0, :].argmin()), 12)
        self.assertLess(float(speed[..., 0, 10]), MAX_SPEED)
        self.assertGreater(float(speed[..., 0, 16]), float(speed[..., 0, 12]))
        self.assertLess(float(speed[..., 0, 24]), float(speed[..., 0, 20]))
        self.assertGreaterEqual(float(acceleration.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(acceleration.max()), MAX_ACCEL + 1e-5)

    def test_pointwise_nominal_risk_finds_crossing_without_scene_label(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            physical_speed_caps=True,
        )).eval()
        with torch.no_grad():
            output = model(state)
            nominal = torch.full_like(output["speed"], MAX_SPEED)
            risk = pointwise_proximity_risk(
                output["path_world"], nominal, output["pickup_dwell"], state,
                proximity_margin=0.25, measured_executor_timing=True,
            )
        self.assertEqual(risk.shape, (1, 1, 2, PATH_POINTS))
        self.assertGreater(float(risk[..., 8:-8].max()), float(risk[..., 0].max()))
        self.assertGreater(float(risk.max()), 0.1)

    def test_physical_caps_localization_penalizes_safe_requests_more(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            physical_speed_caps=True,
        )).eval()
        output = model(state)
        with torch.no_grad():
            risk = pointwise_proximity_risk(
                output["path_world"], torch.full_like(output["speed"], MAX_SPEED),
                output["pickup_dwell"], state, proximity_margin=0.25,
                measured_executor_timing=True,
            )
            high = int(risk[0, 0, 1].argmax())
            low = int(risk[0, 0, 1].argmin())
        safe_output = dict(output)
        risky_output = dict(output)
        safe_request = torch.zeros_like(output["slowdown_request"])
        risky_request = torch.zeros_like(output["slowdown_request"])
        safe_request[0, 0, 1, low] = 1.0
        risky_request[0, 0, 1, high] = 1.0
        safe_output["slowdown_request"] = safe_request
        risky_output["slowdown_request"] = risky_request
        safe_loss = compute_c2_auxiliary_loss(
            safe_output, state, proximity_margin=0.25,
            measured_executor_timing=True,
        )["unnecessary_slow"]
        risky_loss = compute_c2_auxiliary_loss(
            risky_output, state, proximity_margin=0.25,
            measured_executor_timing=True,
        )["unnecessary_slow"]
        self.assertGreater(float(safe_loss), float(risky_loss))

    def test_physical_point_caps_checkpoint_round_trip_is_bit_exact(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            physical_speed_caps=True, waypoint_smoothing_passes=4,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c5_pointcaps.pth"
            save_c2_checkpoint(path, model, step=23)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertTrue(payload["model_config"]["physical_speed_caps"])
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_smooth_depth_retains_gradient_below_zero_for_both_windows(self):
        state = make_state(batch=1)
        cases = (
            ({"slowdown_window": True}, (122, 125)),
            ({"conflict_window": True}, (121, 123)),
        )
        for mode, depth_indices in cases:
            with self.subTest(mode=mode):
                model = JointCoordinator(c2_config(
                    direct_speed_profile=True, mlp_backbone=True,
                    direct_waypoints=True, smooth_depth=True, **mode,
                ))
                with torch.no_grad():
                    model.trajectory_head[-1].bias[list(depth_indices)] = -1.0
                output = model(state)
                self.assertTrue((output["slowdown_window"][..., -1] > 0.0).all())
                self.assertTrue((output["speed"].amin(dim=-1) < MAX_SPEED).all())
                output["speed"].mean().backward()
                grad = model.trajectory_head[-1].bias.grad[list(depth_indices)]
                self.assertTrue((grad.abs() > 0.0).all())

    def test_proximity_collision_is_smooth_and_reaches_free_window_head(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            waypoint_smoothing_passes=8, waypoint_distance_scaling=True,
        ))
        output = model(state)
        metrics = candidate_costs(output, state, proximity_approach_beta=1.0)
        wider_metrics = candidate_costs(
            output, state, proximity_approach_beta=1.0,
            proximity_margin=0.25,
        )
        proximity = metrics["proximity_collision_steps"]
        self.assertEqual(proximity.shape[-1], 96)
        self.assertTrue(torch.isfinite(proximity).all())
        self.assertGreater(float(proximity.mean()), 0.0)
        self.assertGreater(
            float(wider_metrics["proximity_collision_steps"].mean()),
            float(proximity.mean()),
        )

        losses = compute_c2_auxiliary_loss(
            output, state, collision_focus_steps=8,
            proximity_collision=True, proximity_approach_beta=1.0,
            future_collision_coef=5.0, extra_delay_coef=1.0,
            detour_delay_coef=3.0, path_residual_coef=1.0,
        )
        self.assertEqual(float(losses["proximity_collision_mode"]), 1.0)
        self.assertGreater(float(losses["proximity_collision"]), 0.0)
        losses["total"].backward()
        grad = model.trajectory_head[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_conflict_window_ends_at_crossing_and_round_trips(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, conflict_window=True,
            waypoint_smoothing_passes=8, waypoint_distance_scaling=True,
        ))
        policy = CoordinatorActorCritic(model)
        self.assertEqual(policy.action_dim, C2_CONFLICT_WINDOW_ACTION_DIM)
        with torch.no_grad():
            # Last value is agent-1 slowdown depth. The analytic prior itself
            # remains straight and full speed because all final biases are 0.
            model.trajectory_head[-1].bias[-1] = 2.0
            before = model(state)
        window = before["slowdown_window"]
        self.assertEqual(window.shape, (2, 1, 2, 4))
        start, crossing, length, _ = window.unbind(dim=-1)
        self.assertTrue(torch.allclose(start + length, crossing, atol=1e-6))
        speed = before["speed"][..., 1, :]
        self.assertTrue((speed < MAX_SPEED).any())
        # The crossing point and the complete suffix are nominal speed.
        path = before["path_world"][..., 1, :, :]
        ds = (path[..., 1:, :] - path[..., :-1, :]).norm(dim=-1)
        arc = torch.cat((torch.zeros_like(ds[..., :1]), ds.cumsum(dim=-1)), -1)
        after_crossing = arc >= crossing[..., 1, None]
        self.assertTrue(torch.equal(
            speed[after_crossing], torch.full_like(speed[after_crossing], MAX_SPEED)
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c3_crosswin.pth"
            save_c2_checkpoint(path, model, step=7)
            loaded, payload = load_c2_checkpoint(path)
            with torch.no_grad():
                after = loaded(state)
        self.assertEqual(payload["action_dim"], C2_CONFLICT_WINDOW_ACTION_DIM)
        self.assertEqual(payload["accel_knots"], 2)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_conflict_minimum_is_at_crossing_then_recovers(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, conflict_window=True,
            conflict_min_at_crossing=True, smooth_depth=True,
            waypoint_smoothing_passes=8, waypoint_distance_scaling=True,
        ))
        with torch.no_grad():
            # Agent 1 gets an unambiguous slowdown while the path prior stays
            # straight, making the geometric conflict point deterministic.
            model.trajectory_head[-1].bias[-1] = 3.0
            output = model(state)
        path = output["path_world"][0, 0, 1]
        speed = output["speed"][0, 0, 1]
        start, crossing, _, depth = output["slowdown_window"][0, 0, 1]
        ds = (path[1:] - path[:-1]).norm(dim=-1)
        arc = torch.cat((torch.zeros_like(ds[:1]), ds.cumsum(dim=-1)), -1)
        crossing_index = (arc - crossing).abs().argmin()
        self.assertTrue(torch.allclose(
            speed[crossing_index], MAX_SPEED - depth, atol=1e-6
        ))
        self.assertEqual(int(speed.argmin()), int(crossing_index))
        self.assertTrue(torch.equal(
            speed[arc <= start], torch.full_like(speed[arc <= start], MAX_SPEED)
        ))
        recovery = (MAX_SPEED ** 2 - (MAX_SPEED - depth).square()) / (
            2.0 * MAX_ACCEL
        )
        recovered = arc >= crossing + recovery
        self.assertTrue(recovered.any())
        self.assertTrue(torch.equal(
            speed[recovered], torch.full_like(speed[recovered], MAX_SPEED)
        ))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "c3b_crosswin.pth"
            save_c2_checkpoint(checkpoint, model, step=3)
            loaded, payload = load_c2_checkpoint(checkpoint)
        self.assertTrue(payload["model_config"]["conflict_min_at_crossing"])
        with torch.no_grad():
            loaded_output = loaded(state)
        for key in output:
            self.assertTrue(torch.equal(output[key], loaded_output[key]), key)

    def test_conflict_plateau_holds_minimum_until_crossing(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, conflict_window=True,
            conflict_plateau=True, smooth_depth=True,
            waypoint_smoothing_passes=8, waypoint_distance_scaling=True,
        ))
        with torch.no_grad():
            # Agent 1: half of its pre-crossing route is a clear plateau.
            model.trajectory_head[-1].bias[-2] = 0.0
            model.trajectory_head[-1].bias[-1] = 3.0
            output = model(state)
        path = output["path_world"][0, 0, 1]
        speed = output["speed"][0, 0, 1]
        start, crossing, length, depth = output["slowdown_window"][0, 0, 1]
        ds = (path[1:] - path[:-1]).norm(dim=-1)
        arc = torch.cat((torch.zeros_like(ds[:1]), ds.cumsum(dim=-1)), -1)
        plateau = (arc >= start) & (arc <= crossing)
        self.assertGreater(int(plateau.sum()), 1)
        self.assertTrue(torch.allclose(
            speed[plateau], torch.full_like(speed[plateau], MAX_SPEED - depth),
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(start + length, crossing, atol=1e-6))
        self.assertGreater(float(speed[arc < start].max()), float(MAX_SPEED - depth))
        self.assertGreater(float(speed[arc > crossing].max()), float(MAX_SPEED - depth))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "c6_plateau.pth"
            save_c2_checkpoint(checkpoint, model, step=6)
            loaded, payload = load_c2_checkpoint(checkpoint)
        self.assertTrue(payload["model_config"]["conflict_plateau"])
        with torch.no_grad():
            loaded_output = loaded(state)
        for key in output:
            self.assertTrue(torch.equal(output[key], loaded_output[key]), key)

    def test_random_priority_is_straight_and_full_speed_only_for_selected_agent(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        )).eval()
        output = model(state)
        output["path_world"] = output["path_world"].clone()
        output["path_world"][..., 1:16, 1] += 0.4
        output["speed"] = torch.full_like(output["speed"], MIN_SPEED)
        priority = torch.tensor([0, 1], dtype=torch.long)
        constrained = apply_fixed_priority(output, state, priority)

        self.assertTrue(torch.equal(
            constrained["speed"][0, :, 0],
            torch.full_like(constrained["speed"][0, :, 0], MAX_SPEED),
        ))
        self.assertTrue(torch.equal(
            constrained["speed"][1, :, 1],
            torch.full_like(constrained["speed"][1, :, 1], MAX_SPEED),
        ))
        self.assertTrue(torch.equal(
            constrained["speed"][0, :, 1], output["speed"][0, :, 1]
        ))
        self.assertTrue(torch.equal(
            constrained["speed"][1, :, 0], output["speed"][1, :, 0]
        ))
        midpoint = 0.5 * (state.root_xy + state.box_xyz[..., :2])
        self.assertTrue(torch.allclose(
            constrained["path_world"][0, 0, 0, 8], midpoint[0, 0]
        ))
        self.assertTrue(torch.allclose(
            constrained["path_world"][1, 0, 1, 8], midpoint[1, 1]
        ))
        self.assertTrue(torch.equal(
            constrained["path_world"][0, :, 1], output["path_world"][0, :, 1]
        ))

    def test_slowdown_window_is_joint_smooth_bounded_output(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            slowdown_width_max=3.0,
        ))
        policy = CoordinatorActorCritic(model)
        self.assertEqual(policy.action_dim, C2_SLOWDOWN_WINDOW_ACTION_DIM)
        with torch.no_grad():
            # The final six values are agent-wise centre/width/depth.
            model.trajectory_head[-1].bias[-1] = 2.0
            output = model(state)
        self.assertEqual(output["trajectory"].shape, (2, 1, 2, PATH_POINTS, 3))
        self.assertEqual(output["slowdown_window"].shape, (2, 1, 2, 3))
        self.assertTrue((output["slowdown_window"][..., 1] <= 3.0 + 1e-6).all())
        speed = output["speed"][..., 1, :]
        self.assertTrue((speed < MAX_SPEED).any())
        self.assertTrue(torch.equal(speed[..., 0], torch.full_like(speed[..., 0], MAX_SPEED)))
        self.assertTrue(torch.equal(speed[..., -1], torch.full_like(speed[..., -1], MAX_SPEED)))

    def test_extra_delay_is_single_speed_cost_and_reaches_window(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            slowdown_width_max=3.0,
        ))
        with torch.no_grad():
            model.trajectory_head[-1].bias[-1] = 2.0
        output = apply_fixed_priority(model(state), state, torch.zeros(2, dtype=torch.long))
        losses = compute_c2_auxiliary_loss(
            output, state,
            future_collision_coef=0.0,
            speed_smooth_coef=0.0,
            unnecessary_slow_coef=0.0,
            speed_efficiency_coef=0.0,
            path_residual_coef=0.0,
            path_smooth_coef=0.0,
            extra_delay_coef=1.0,
            priority_agent=torch.zeros(2, dtype=torch.long),
            measured_executor_timing=True,
        )
        self.assertGreater(float(losses["extra_delay"]), 0.0)
        self.assertAlmostEqual(
            float(losses["total"]), float(losses["extra_delay"]), places=6
        )
        losses["total"].backward()
        self.assertGreater(
            float(model.trajectory_head[-1].bias.grad[-1].abs()), 0.0
        )

    def test_detour_delay_prices_only_yielder_extra_path_time(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            slowdown_width_max=0.0, waypoint_smoothing_passes=8,
        ))
        with torch.no_grad():
            # Bend every non-anchor y waypoint of agent 1 while leaving speed nominal.
            model.trajectory_head[-1].bias[61:120:2] = 0.5
        priority = torch.zeros(1, dtype=torch.long)
        output = apply_fixed_priority(model(state), state, priority)
        losses = compute_c2_auxiliary_loss(
            output, state, future_collision_coef=0.0,
            speed_smooth_coef=0.0, unnecessary_slow_coef=0.0,
            speed_efficiency_coef=0.0, extra_delay_coef=0.0,
            detour_delay_coef=3.0, path_residual_coef=0.0,
            path_smooth_coef=0.0, priority_agent=priority,
            measured_executor_timing=True,
        )
        self.assertGreater(float(losses["detour_delay"]), 0.0)
        self.assertAlmostEqual(
            float(losses["total"]), 3.0 * float(losses["detour_delay"]),
            places=5,
        )
        losses["total"].backward()
        self.assertGreater(
            float(model.trajectory_head[-1].bias.grad[61:120].abs().sum()), 0.0
        )

    def test_slowdown_window_checkpoint_round_trip_is_bit_exact(self):
        torch.manual_seed(17)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            slowdown_width_max=3.0, waypoint_smoothing_passes=8,
            waypoint_distance_scaling=True,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2_slowdown_window.pth"
            save_c2_checkpoint(path, model, step=23)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["action_dim"], C2_SLOWDOWN_WINDOW_ACTION_DIM)
        self.assertTrue(payload["model_config"]["slowdown_window"])
        self.assertEqual(payload["model_config"]["slowdown_width_max"], 3.0)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_explicit_speed_losses_force_assigned_yielder_and_box_anchor(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, slowdown_window=True,
            slowdown_width_max=0.0,
        ))
        with torch.no_grad():
            # Give both counterfactual yield slots a visible slowdown window.
            model.trajectory_head[-1].bias[-1] = 2.0
            model.trajectory_head[-1].bias[-4] = 2.0
        priority = torch.tensor([0, 1], dtype=torch.long)
        output = apply_fixed_priority(model(state), state, priority)
        losses = compute_c2_auxiliary_loss(
            output, state, future_collision_coef=0.0,
            speed_smooth_coef=0.0, unnecessary_slow_coef=0.0,
            speed_efficiency_coef=0.0, extra_delay_coef=0.0,
            path_residual_coef=0.0, path_smooth_coef=0.0,
            priority_agent=priority,
            explicit_gap_coef=10.0,
            explicit_anchor_coef=3.0,
            explicit_post_coef=3.0,
            initial_plan_mask=torch.tensor([True, False]),
            initial_plan_weight=3.0,
            measured_executor_timing=True,
        )
        self.assertGreater(float(losses["explicit_need_rate"]), 0.0)
        self.assertGreater(float(losses["explicit_anchor"]), 0.0)
        self.assertGreater(float(losses["explicit_post"]), 0.0)
        self.assertEqual(float(losses["initial_plan_weight"]), 3.0)
        losses["total"].backward()
        self.assertGreater(
            float(model.trajectory_head[-1].bias.grad.abs().sum()), 0.0
        )

    def test_time_shifted_consistency_is_zero_for_identical_plan(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        )).eval()
        output = model(state)
        target = build_trajectory_consistency_target(
            output, state, state, 0.0,
            torch.ones(state.batch_size, dtype=torch.bool),
            measured_executor_timing=True,
        )
        losses = trajectory_consistency_loss(
            output, state, target, measured_executor_timing=True
        )
        self.assertAlmostEqual(float(losses["total"]), 0.0, places=7)
        self.assertGreater(float(losses["valid_fraction"]), 0.0)

    def test_consistency_penalizes_changed_joint_trajectory(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        previous = model(state)
        target = build_trajectory_consistency_target(
            previous, state, state, 0.0,
            torch.ones(state.batch_size, dtype=torch.bool),
            measured_executor_timing=True,
        )
        with torch.no_grad():
            model.trajectory_head[-1].bias[0] = 0.5
            model.trajectory_head[-1].bias[-1] = -1.0
        output = model(state)
        losses = compute_c2_auxiliary_loss(
            output, state,
            measured_executor_timing=True,
            consistency_target=target,
            consistency_coef=1.0,
        )
        self.assertGreater(float(losses["consistency"]), 0.0)
        self.assertEqual(float(losses["consistency_coef"]), 1.0)
        losses["total"].backward()
        self.assertGreater(
            float(model.trajectory_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_consistency_masks_phase_change(self):
        previous_state = make_state(batch=1)
        current_state = previous_state.clone()
        current_state.phase.fill_(1.0)
        output = JointCoordinator(c2_config()).eval()(previous_state)
        target = build_trajectory_consistency_target(
            output, previous_state, current_state, 0.2,
            torch.ones(1, dtype=torch.bool),
        )
        losses = trajectory_consistency_loss(output, current_state, target)
        self.assertEqual(float(target["valid"].float().sum()), 0.0)
        self.assertEqual(float(losses["total"]), 0.0)

    def test_waypoint_filter_reduces_pointwise_zigzag_and_keeps_anchors(self):
        state = make_state(batch=1)
        _, frame = state_to_tokens(state)
        residual = torch.zeros(1, 1, 2, WAYPOINT_RESIDUAL_POINTS, 2)
        residual[..., 1::2, 1] = 0.20
        residual[..., 0::2, 1] = -0.20
        raw = build_waypoint_paths(frame, residual)
        smooth = build_waypoint_paths(frame, residual, smoothing_passes=4)
        raw_d2 = raw[..., 2:, :] - 2.0 * raw[..., 1:-1, :] + raw[..., :-2, :]
        smooth_d2 = (
            smooth[..., 2:, :] - 2.0 * smooth[..., 1:-1, :]
            + smooth[..., :-2, :]
        )
        self.assertLess(float(smooth_d2.square().mean()), float(raw_d2.square().mean()))
        self.assertTrue(torch.equal(smooth[:, 0, :, 0], frame["root"]))
        self.assertTrue(torch.equal(smooth[:, 0, :, 16], frame["box"]))
        self.assertTrue(torch.equal(smooth[:, 0, :, 32], frame["goal"]))

    def test_waypoint_distance_scaling_shrinks_offsets_near_goal(self):
        state = make_state(batch=1)
        _, frame = state_to_tokens(state)
        frame = dict(frame)
        frame["goal"] = frame["box"] + torch.tensor([[[0.25, 0.0]]])
        residual = torch.zeros(1, 1, 2, WAYPOINT_RESIDUAL_POINTS, 2)
        residual[..., 15:, 1] = 0.20
        baseline = build_waypoint_paths(frame, torch.zeros_like(residual))
        raw = build_waypoint_paths(frame, residual)
        scaled = build_waypoint_paths(frame, residual, distance_scaling=True)
        raw_offset = (raw[..., 17:32, :] - baseline[..., 17:32, :]).norm(dim=-1)
        scaled_offset = (
            scaled[..., 17:32, :] - baseline[..., 17:32, :]
        ).norm(dim=-1)
        self.assertTrue(torch.allclose(scaled_offset, 0.25 * raw_offset))

    def test_k1_anchors_speed_and_fixed_dwell(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config()).eval()
        with torch.inference_mode():
            output = model(state)
        self.assertEqual(output["path_world"].shape, (2, 1, 2, PATH_POINTS, 2))
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, 0], state.root_xy))
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, 16], state.box_xyz[..., :2]))
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, 32], state.goal_xy))
        self.assertTrue(torch.equal(output["speed"], torch.full_like(output["speed"], MAX_SPEED)))
        self.assertTrue(torch.equal(output["pickup_dwell"], torch.full_like(output["pickup_dwell"], 1.5)))

    def test_policy_has_no_candidate_or_dwell_action(self):
        state = make_state(batch=2)
        policy = CoordinatorActorCritic(JointCoordinator(c2_config()))
        _, action, log_prob, value = policy.act(state)
        self.assertEqual(policy.action_dim, C2_ACTION_DIM)
        self.assertEqual(action.shape, (2, C2_ACTION_DIM))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(value.shape, (2,))

    def test_actual_speed_mode_continues_replan_speed(self):
        state = make_state(batch=2)
        expected = state.root_vel_xy.norm(dim=-1)
        model = JointCoordinator(c2_config(actual_initial_speed=True)).eval()
        with torch.inference_mode():
            output = model(state)
        self.assertTrue(torch.allclose(output["speed"][:, 0, :, 0], expected))

    def test_immediate_mode_controls_current_command(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(immediate_slowdown=True)).eval()
        with torch.no_grad():
            model.accel_head[-1].bias[0] = -1.0
            model.accel_head[-1].bias[8] = 1.0
            output = model(state)
        current = output["speed"][:, 0, :, 0]
        self.assertTrue((current[:, 0] < MAX_SPEED).all())
        self.assertTrue(torch.equal(current[:, 1], torch.full_like(current[:, 1], MAX_SPEED)))

    def test_direct_speed_knots_localize_slowdown_and_recover(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(direct_speed_profile=True)).eval()
        with torch.no_grad():
            model.accel_head[-1].bias.zero_()
            model.accel_head[-1].bias[8 + 3:8 + 5] = 1.0
            output = model(state)
        profile = output["speed"][0, 0, 1]
        self.assertEqual(float(profile[0]), MAX_SPEED)
        self.assertLess(float(profile[16]), MAX_SPEED)
        self.assertEqual(float(profile[-1]), MAX_SPEED)

    def test_future_collision_loss_reaches_path_and_speed_heads(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config())
        losses = compute_c2_auxiliary_loss(model(state), state)
        self.assertGreater(float(losses["future_collision"]), 0.0)
        self.assertEqual(float(losses["nominal_conflict_rate"]), 1.0)
        losses["total"].backward()
        for name in ("path_head", "accel_head"):
            head = getattr(model, name)
            grad = sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
            self.assertGreater(grad, 0.0, name)

    def test_peak_mode_does_not_dilute_brief_crossing(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config()).eval()
        output = model(state)
        mean_loss = compute_c2_auxiliary_loss(output, state)
        peak_loss = compute_c2_auxiliary_loss(output, state, peak_collision=True)
        self.assertGreater(
            float(peak_loss["future_collision"]),
            float(mean_loss["future_collision"]),
        )
        self.assertEqual(float(peak_loss["peak_collision_mode"]), 1.0)

    def test_topk_collision_focuses_crossing_without_speed_labels(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        output = model(state)
        mean_loss = compute_c2_auxiliary_loss(output, state)
        focused = compute_c2_auxiliary_loss(
            output, state,
            collision_focus_steps=8,
            speed_efficiency_coef=0.1,
            path_residual_coef=1.0,
            path_smooth_coef=10.0,
        )
        self.assertGreater(
            float(focused["future_collision"]),
            float(mean_loss["future_collision"]),
        )
        self.assertEqual(float(focused["collision_focus_steps"]), 8.0)
        self.assertEqual(float(focused["fixed_yield_speed"]), 0.0)
        self.assertEqual(float(focused["state_ordered_target_gap"]), 0.0)
        self.assertGreaterEqual(float(focused["speed_efficiency"]), 0.0)
        focused["total"].backward()
        self.assertGreater(
            float(model.trajectory_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_topk_collision_rejects_too_many_samples(self):
        state = make_state(batch=1)
        output = JointCoordinator(c2_config()).eval()(state)
        with self.assertRaises(ValueError):
            compute_c2_auxiliary_loss(
                output, state, collision_focus_steps=97
            )

    def test_time_uncertainty_expands_future_collision_window(self):
        state = make_state(batch=2)
        output = JointCoordinator(c2_config()).eval()(state)
        exact = compute_c2_auxiliary_loss(
            output, state, collision_focus_steps=8
        )
        robust = compute_c2_auxiliary_loss(
            output, state,
            collision_focus_steps=8,
            collision_time_uncertainty=0.75,
        )
        self.assertGreaterEqual(
            float(robust["future_collision"]),
            float(exact["future_collision"]),
        )
        self.assertEqual(float(robust["collision_time_uncertainty"]), 0.75)

    def test_measured_executor_timing_reaches_joint_trajectory_head(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            collision_focus_steps=8,
            measured_executor_timing=True,
        )
        self.assertEqual(float(losses["measured_executor_timing"]), 1.0)
        losses["total"].backward()
        grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.trajectory_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad, 0.0)

    def test_negative_time_uncertainty_is_rejected(self):
        state = make_state(batch=1)
        output = JointCoordinator(c2_config()).eval()(state)
        with self.assertRaises(ValueError):
            compute_c2_auxiliary_loss(
                output, state, collision_time_uncertainty=-0.1
            )

    def test_future_collision_coefficient_scales_only_collision_term(self):
        state = make_state(batch=2)
        output = JointCoordinator(c2_config()).eval()(state)
        low = compute_c2_auxiliary_loss(
            output, state, future_collision_coef=10.0
        )
        high = compute_c2_auxiliary_loss(
            output, state, future_collision_coef=50.0
        )
        expected = 40.0 * low["future_collision"]
        self.assertTrue(torch.allclose(high["total"] - low["total"], expected))
        self.assertEqual(float(low["future_collision_coef"]), 10.0)

    def test_larger_clearance_strengthens_crossing_loss(self):
        state = make_state(batch=2)
        output = JointCoordinator(c2_config()).eval()(state)
        base = compute_c2_auxiliary_loss(output, state, human_clearance=1.0)
        margin = compute_c2_auxiliary_loss(output, state, human_clearance=1.5)
        self.assertGreater(float(margin["future_collision"]), float(base["future_collision"]))
        self.assertEqual(float(margin["human_clearance"]), 1.5)

    def test_arrival_gap_breaks_tie_toward_one_yielding_agent(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(immediate_slowdown=True))
        output = model(state)
        losses = compute_c2_auxiliary_loss(output, state, arrival_gap=True)
        self.assertGreater(float(losses["arrival_gap_loss"]), 0.0)
        self.assertEqual(float(losses["arrival_gap_mode"]), 1.0)
        losses["total"].backward()
        grad = model.accel_head[-1].bias.grad.reshape(2, -1)[:, 0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_fixed_a1_priority_penalizes_a0_slowdown(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(immediate_slowdown=True))
        output = model(state)
        losses = compute_c2_auxiliary_loss(
            output, state, arrival_gap=True, fixed_yield_agent1=True
        )
        self.assertEqual(float(losses["fixed_yield_agent1"]), 1.0)
        self.assertGreaterEqual(float(losses["priority_speed"]), 0.0)

    def test_fixed_priority_leaves_existing_large_gap_alone(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(immediate_slowdown=True)).eval()
        output = model(state)
        output["pickup_dwell"] = output["pickup_dwell"].clone()
        output["pickup_dwell"][:, :, 0] += 3.0
        losses = compute_c2_auxiliary_loss(
            output, state, arrival_gap=True, fixed_yield_agent1=True
        )
        self.assertEqual(float(losses["arrival_gap_loss"]), 0.0)

    def test_bidirectional_target_gap_chooses_one_yielder(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(direct_speed_profile=True))
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            bidirectional_target_gap=True, arrival_gap_coef=10.0,
        )
        self.assertEqual(float(losses["bidirectional_target_gap"]), 1.0)
        self.assertGreater(float(losses["target_gap_need_rate"]), 0.0)
        self.assertGreater(float(losses["bidirectional_gap_loss"]), 0.0)
        losses["total"].backward()
        grad = model.accel_head[-1].bias.grad
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_state_ordered_gap_changes_yielder_with_nominal_arrival_order(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(direct_speed_profile=True))
        output = model(state)
        losses = compute_c2_auxiliary_loss(
            output, state,
            state_ordered_target_gap=True, arrival_gap_coef=10.0,
        )
        self.assertEqual(float(losses["state_ordered_target_gap"]), 1.0)
        self.assertGreater(float(losses["target_gap_need_rate"]), 0.0)
        self.assertGreater(float(losses["prefix_speed_target_loss"]), 0.0)
        self.assertGreaterEqual(float(losses["restore_speed_loss"]), 0.0)
        self.assertGreater(float(losses["target_yield_speed"]), 0.0)
        losses["total"].backward()
        grad = model.accel_head[-1].bias.grad.reshape(2, -1)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_fixed_yield_speed_gives_clear_slow_target(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            state_ordered_target_gap=True,
            fixed_yield_speed=MIN_SPEED,
            path_collision_only=True,
            arrival_gap_coef=10.0,
        )
        self.assertEqual(float(losses["fixed_yield_speed"]), MIN_SPEED)
        self.assertEqual(float(losses["path_collision_only"]), 1.0)
        self.assertAlmostEqual(float(losses["target_yield_speed"]), MIN_SPEED, places=5)
        losses["total"].backward()
        self.assertGreater(float(model.trajectory_head[-1].weight.grad.abs().sum()), 0.0)

    def test_fixed_yield_speed_rejects_out_of_contract_value(self):
        state = make_state(batch=1)
        output = JointCoordinator(c2_config()).eval()(state)
        with self.assertRaises(ValueError):
            compute_c2_auxiliary_loss(
                output, state, state_ordered_target_gap=True,
                fixed_yield_speed=0.2,
            )

    def test_arrival_features_preserve_output_contract(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, arrival_features=True
        )).eval()
        with torch.inference_mode():
            output = model(state)
        self.assertEqual(output["speed"].shape, (2, 1, 2, PATH_POINTS))
        self.assertTrue(torch.isfinite(output["speed"]).all())

    def test_flat_mlp_preserves_c2_contract_without_transformer(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True
        )).eval()
        with torch.inference_mode():
            output = model(state)
        self.assertEqual(output["path_world"].shape, (2, 1, 2, PATH_POINTS, 2))
        self.assertEqual(output["speed"].shape, (2, 1, 2, PATH_POINTS))
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, 0], state.root_xy))
        self.assertTrue(torch.allclose(
            output["path_world"][:, 0, :, 16], state.box_xyz[..., :2]
        ))
        self.assertTrue(torch.allclose(output["path_world"][:, 0, :, 32], state.goal_xy))
        self.assertFalse(any(
            isinstance(module, (torch.nn.TransformerEncoder, torch.nn.TransformerDecoder))
            for module in model.modules()
        ))

    def test_flat_mlp_checkpoint_round_trip_is_bit_exact(self):
        torch.manual_seed(9)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2_mlp.pth"
            save_c2_checkpoint(path, model, step=13)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["step"], 13)
        self.assertTrue(payload["model_config"]["mlp_backbone"])
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_mlp_conflict_features_keep_state_only_contract(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            mlp_conflict_features=True,
        )).eval()
        self.assertEqual(
            model.mlp_backbone[0].in_features,
            2 * 3 * 12 + MLP_CONFLICT_FEATURES,
        )
        with torch.inference_mode():
            output = model(state)
        self.assertEqual(output["trajectory"].shape, (2, 1, 2, PATH_POINTS, 3))
        self.assertTrue(torch.isfinite(output["trajectory"]).all())

    def test_mlp_conflict_feature_checkpoint_is_bit_exact(self):
        torch.manual_seed(12)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            mlp_conflict_features=True,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2_conflict_features.pth"
            save_c2_checkpoint(path, model, step=23)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertTrue(payload["model_config"]["mlp_conflict_features"])
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_direct_waypoints_are_independent_baseline_residuals(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True,
        )).eval()
        with torch.no_grad():
            straight = model(state)
            model.path_head[-1].bias[0] = 0.5
            bent = model(state)
        self.assertEqual(
            bent["waypoint_residual"].shape,
            (1, 1, 2, WAYPOINT_RESIDUAL_POINTS, 2),
        )
        self.assertFalse(torch.equal(
            bent["path_local"][..., 1, :], straight["path_local"][..., 1, :]
        ))
        # Delta at P1 is relative to P1_base, not an increment propagated to P2.
        self.assertTrue(torch.equal(
            bent["path_local"][..., 2, :], straight["path_local"][..., 2, :]
        ))
        self.assertTrue(torch.equal(bent["path_world"][:, 0, :, 0], state.root_xy))
        self.assertTrue(torch.equal(
            bent["path_world"][:, 0, :, 16], state.box_xyz[..., :2]
        ))
        self.assertTrue(torch.equal(bent["path_world"][:, 0, :, 32], state.goal_xy))

    def test_direct_waypoint_policy_action_and_held_approach_contract(self):
        state = make_state(batch=2, held=True)
        policy = CoordinatorActorCritic(JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True,
        )))
        output, action, log_prob, value = policy.act(state)
        self.assertEqual(policy.action_dim, C2_WAYPOINT_ACTION_DIM)
        self.assertEqual(action.shape, (2, C2_WAYPOINT_ACTION_DIM))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.equal(
            output["waypoint_residual"][..., :15, :],
            torch.zeros_like(output["waypoint_residual"][..., :15, :]),
        ))
        path_actions = 2 * WAYPOINT_RESIDUAL_POINTS * 2
        std = policy.action_log_std.exp()
        self.assertTrue(torch.allclose(
            std[:path_actions], torch.full_like(std[:path_actions], 0.005)
        ))
        self.assertTrue(torch.allclose(
            std[path_actions:], torch.full_like(std[path_actions:], 0.15)
        ))
        distribution, _, _ = policy.distribution(state)
        self.assertTrue(torch.allclose(
            distribution.stddev[0, :path_actions],
            torch.full_like(distribution.stddev[0, :path_actions], 0.005),
        ))

    def test_direct_waypoint_checkpoint_round_trip_is_bit_exact(self):
        torch.manual_seed(10)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2_waypoint.pth"
            save_c2_checkpoint(path, model, step=17)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["action_dim"], C2_WAYPOINT_ACTION_DIM)
        self.assertTrue(payload["model_config"]["direct_waypoints"])
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_direct_waypoint_regularizers_reach_path_head(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True,
        ))
        with torch.no_grad():
            model.path_head[-1].bias[0] = 0.5
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            state_ordered_target_gap=True, arrival_gap_coef=10.0,
            path_residual_coef=1.0, path_smooth_coef=0.1,
        )
        self.assertGreater(float(losses["path_residual"]), 0.0)
        self.assertGreater(float(losses["path_smooth"]), 0.0)
        losses["total"].backward()
        self.assertGreater(float(model.path_head[-1].bias.grad.abs().sum()), 0.0)

    def test_direct_waypoint_future_collision_reaches_path_head(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True,
        ))
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            state_ordered_target_gap=True, arrival_gap_coef=10.0,
        )
        self.assertGreater(float(losses["future_collision"]), 0.0)
        losses["total"].backward()
        self.assertGreater(float(model.path_head[-1].bias.grad.abs().sum()), 0.0)

    def test_joint_point_speed_is_one_spatiotemporal_trajectory(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            waypoint_smoothing_passes=4,
        )).eval()
        policy = CoordinatorActorCritic(model)
        with torch.no_grad():
            output, action, _, _ = policy.act(state, deterministic=True)
        self.assertEqual(action.shape, (2, C2_JOINT_POINT_ACTION_DIM))
        self.assertEqual(output["trajectory"].shape, (2, 1, 2, PATH_POINTS, 3))
        self.assertTrue(torch.equal(
            output["trajectory"][..., :2], output["path_world"]
        ))
        self.assertTrue(torch.equal(
            output["trajectory"][..., 2], output["speed"]
        ))
        self.assertTrue(torch.equal(
            output["speed"], torch.full_like(output["speed"], MAX_SPEED)
        ))
        self.assertIsNone(model.path_head)
        self.assertIsNone(model.accel_head)

    def test_joint_point_speed_collision_gradient_reaches_single_head(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        losses = compute_c2_auxiliary_loss(
            model(state), state,
            state_ordered_target_gap=True, arrival_gap_coef=10.0,
            path_residual_coef=1.0, path_smooth_coef=0.1,
        )
        losses["total"].backward()
        grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.trajectory_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad, 0.0)

    def test_dual_delay_penalizes_two_slow_agents_only(self):
        state = make_state(batch=2)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
        ))
        output = model(state)

        one_slow = dict(output)
        one_slow["speed"] = torch.full_like(output["speed"], MAX_SPEED)
        one_slow["speed"][:, :, 0] = MIN_SPEED
        one = compute_c2_auxiliary_loss(
            one_slow, state, future_collision_coef=0.0,
            dual_delay_coef=1.0,
        )

        speed = torch.full_like(
            output["speed"], 0.5 * (MIN_SPEED + MAX_SPEED),
            requires_grad=True,
        )
        both_slow = dict(output)
        both_slow["speed"] = speed
        both = compute_c2_auxiliary_loss(
            both_slow, state, future_collision_coef=0.0,
            dual_delay_coef=1.0,
        )
        self.assertEqual(float(one["dual_delay"]), 0.0)
        self.assertGreater(float(both["dual_delay"]), 0.0)
        both["total"].backward()
        self.assertGreater(float(speed.grad.abs().sum()), 0.0)

    def test_joint_point_speed_checkpoint_round_trip_is_bit_exact(self):
        torch.manual_seed(11)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True,
            direct_waypoints=True, joint_point_speed=True,
            waypoint_smoothing_passes=4,
        )).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2_joint_point.pth"
            save_c2_checkpoint(path, model, step=19)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["action_dim"], C2_JOINT_POINT_ACTION_DIM)
        self.assertEqual(payload["accel_knots"], PATH_POINTS)
        self.assertEqual(payload["model_config"]["waypoint_smoothing_passes"], 4)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_path_residual_loss_prefers_straight_carry_leg(self):
        state = make_state(batch=1)
        model = JointCoordinator(c2_config(
            direct_speed_profile=True, mlp_backbone=True
        ))
        with torch.no_grad():
            model.path_head[-1].bias[-1] = 0.5
        losses = compute_c2_auxiliary_loss(
            model(state), state, state_ordered_target_gap=True,
            arrival_gap_coef=10.0, path_residual_coef=1.0,
        )
        self.assertGreater(float(losses["path_residual"]), 0.0)
        self.assertEqual(float(losses["path_residual_coef"]), 1.0)
        losses["total"].backward()
        grad = model.path_head[-1].bias.grad
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_checkpoint_round_trip_is_bit_exact(self):
        torch.manual_seed(7)
        state = make_state(batch=1)
        model = JointCoordinator(c2_config()).eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c2.pth"
            save_c2_checkpoint(path, model, step=11)
            loaded, payload = load_c2_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
            self.assertEqual(payload["step"], 11)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)


if __name__ == "__main__":
    unittest.main()
