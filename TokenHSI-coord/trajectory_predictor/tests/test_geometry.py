import math
import unittest

import torch

from tokenhsi.utils import steer_path as sp
from trajectory_predictor.dataset import sample_states
from trajectory_predictor.geometry import (
    augment_state_and_path,
    baseline_path_local,
    path_to_shared,
    path_to_world,
    resample_coarse,
    state_to_tokens,
)
from trajectory_predictor.model import JointTrajectoryPredictor


class GeometryTest(unittest.TestCase):
    def test_shared_frame_round_trip(self):
        state, _, _ = sample_states(8, 11)
        points = torch.randn(8, 2, 33, 2)
        restored = path_to_world(path_to_shared(points, state), state)
        torch.testing.assert_close(restored, points, atol=2e-6, rtol=0)

    def test_rigid_augmentation_preserves_canonical_input(self):
        state, _, _ = sample_states(8, 12)
        _, frame = state_to_tokens(state)
        baseline = path_to_world(baseline_path_local(frame), state)
        tokens0, _ = state_to_tokens(state)
        transformed_state, transformed_path = augment_state_and_path(state, baseline, seed=19)
        tokens1, _ = state_to_tokens(transformed_state)
        torch.testing.assert_close(tokens1, tokens0, atol=2e-6, rtol=0)
        restored = path_to_shared(transformed_path, transformed_state)
        torch.testing.assert_close(restored, path_to_shared(baseline, state), atol=3e-6, rtol=0)

    def test_hard_anchors_and_dense_shape(self):
        state, _, _ = sample_states(5, 13)
        model = JointTrajectoryPredictor().eval()
        with torch.inference_mode():
            coarse = model(state)["coarse_world"]
        torch.testing.assert_close(coarse[:, :, 0], state.root_xy, atol=2e-6, rtol=0)
        torch.testing.assert_close(coarse[:, :, 16], state.box_xyz[..., :2], atol=2e-6, rtol=0)
        torch.testing.assert_close(coarse[:, :, 32], state.goal_xy, atol=2e-6, rtol=0)
        dense, end_s = resample_coarse(coarse, with_end=True)
        self.assertEqual(tuple(dense.shape), (5, 2, sp.V, 2))
        self.assertEqual(tuple(end_s.shape), (5, 2))


if __name__ == "__main__":
    unittest.main()
