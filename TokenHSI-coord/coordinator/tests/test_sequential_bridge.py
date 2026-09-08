"""CPU regressions for the new executor bridge; no Isaac Gym or GPU needed."""

import tempfile
import unittest
from pathlib import Path

import torch

from coordinator.c2_checkpoint import c2_config, save_c2_checkpoint
from coordinator.model import JointCoordinator
from coordinator.sequential_bridge import (
    carry_rows, load_planner, plan_validity, resample_plan, select_stack_plan,
)
from coordinator.tests.common import make_state


def straight(state, candidates=1):
    t = torch.linspace(0, 1, 17)[None, None, :, None]
    root, box, goal = state.root_xy, state.box_xyz[..., :2], state.goal_xy
    a = root[:, :, None] + t * (box - root)[:, :, None]
    b = box[:, :, None] + t * (goal - box)[:, :, None]
    path = torch.cat((a, b[:, :, 1:]), dim=-2)[:, None].repeat(1, candidates, 1, 1, 1)
    return {"path_world": path, "speed": torch.full(path.shape[:-1], 1.5),
            "pickup_dwell": torch.zeros(state.batch_size, candidates, 2)}


class SequentialBridgeTest(unittest.TestCase):
    def test_phase_ownership_and_rehearsal(self):
        mask = carry_rows(torch.arange(6), torch.zeros(6, dtype=torch.bool))
        self.assertEqual(mask.tolist(), [[True, False], [True, False], [False, False],
                                         [False, True], [False, True], [False, False]])
        self.assertTrue(carry_rows(torch.arange(6), torch.ones(6, dtype=torch.bool)).all())

    def test_resampling_keeps_speed_at_its_spatial_position(self):
        x = torch.linspace(0, 8, 33)
        path = torch.stack((x, torch.zeros_like(x)), dim=-1)[None, None].repeat(1, 2, 1, 1)
        speed = torch.full((1, 2, 33), 1.5)
        speed[:, :, 16] = 0.375  # Local slowdown at x=4m, not at 1.125m.
        dense, v, end = resample_plan(path, speed)
        torch.testing.assert_close(end, torch.full((1, 2), 8.))
        self.assertAlmostEqual(float(v[0, 0, 40]), 0.375, places=5)
        self.assertAlmostEqual(float(v[0, 0, 10]), 1.5, places=5)
        self.assertAlmostEqual(float(v[0, 0, 41]), 0.825, places=5)
        torch.testing.assert_close(dense[0, 0, 80:], path[0, 0, -1].expand(240, 2))
        self.assertTrue((v[0, 0, 80:] == 1.5).all())

    def test_degenerate_and_held_paths_remain_finite(self):
        state = make_state(1, held=True)
        output = straight(state)
        valid = plan_validity(output["path_world"], output["speed"], state,
                              torch.ones(1, 2, dtype=torch.bool))
        self.assertTrue(valid.all())
        state.goal_xy = state.root_xy.clone()
        output = straight(state)
        dense, speed, end = resample_plan(output["path_world"][:, 0], output["speed"][:, 0])
        self.assertTrue(torch.isfinite(dense).all() & torch.isfinite(speed).all())
        self.assertTrue((end == 0).all())

    def test_invalid_fast_candidate_cannot_win(self):
        state = make_state(1)
        output = straight(state, 2)
        output["path_world"][0, 0, 0, 0] += 1  # Broken root anchor.
        selected = select_stack_plan(output, state, torch.tensor([[True, False]]))
        self.assertEqual(selected[-1].item(), 1)
        self.assertTrue(selected[2].item())

    def test_inactive_path_is_ignored_and_stays_at_actual_root(self):
        state = make_state(1)
        output = straight(state)
        output["path_world"][0, 0, 1] = float("nan")
        path, speed, valid, _, _ = select_stack_plan(output, state, torch.tensor([[True, False]]))
        self.assertTrue(valid.item())
        torch.testing.assert_close(path[0, 1], state.root_xy[0, 1].expand(33, 2))
        self.assertTrue(torch.isfinite(speed).all())

    def test_overflow_and_bad_speed_are_rejected(self):
        state = make_state(1)
        state.goal_xy[0, 0, 0] = 100
        output = straight(state)
        self.assertFalse(select_stack_plan(output, state, torch.tensor([[True, False]]))[2].item())
        output = straight(make_state(1))
        output["speed"][0, 0, 0, 10] = -1
        self.assertFalse(select_stack_plan(output, make_state(1), torch.tensor([[True, False]]))[2].item())

    def test_c13_architecture_checkpoint_load_and_inference(self):
        model = JointCoordinator(c2_config(
            mlp_backbone=True, direct_waypoints=True, direct_speed_profile=True,
            joint_point_speed=True, physical_speed_caps=True,
            waypoint_smoothing_passes=8, waypoint_distance_scaling=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c13_synthetic.pth"
            save_c2_checkpoint(path, model, extras={"random_priority": True})
            loaded, payload = load_planner(path, "cpu")
            self.assertTrue(payload["extras"]["random_priority"])
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            result = loaded(make_state(1))
            self.assertEqual(result["trajectory"].shape, (1, 1, 2, 33, 3))
            dense, _, _ = resample_plan(result["path_world"][:, 0], result["speed"][:, 0])
            self.assertEqual(dense.shape, (1, 2, 320, 2))

    def test_executor_checkpoint_is_not_accepted_as_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masteer.pth"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(ValueError, "coordinator checkpoint"):
                load_planner(path, "cpu")
