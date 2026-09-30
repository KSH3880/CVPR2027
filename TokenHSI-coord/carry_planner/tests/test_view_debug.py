import unittest

import numpy as np
import torch

from carry_planner.view_debug import (
    rejected_path_vertices, rejection_reason, viewer_cross_slots,
)


class CarryViewDebugTest(unittest.TestCase):
    def test_path_vertices_preserve_segments_and_height(self):
        path = np.asarray((
            ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)),
            ((3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
        ), dtype=np.float32)
        vertices = rejected_path_vertices(path, height=0.25)
        self.assertEqual(vertices.shape, (2, 2, 6))
        np.testing.assert_allclose(
            vertices[0, 0], (0.0, 0.0, 0.25, 1.0, 0.0, 0.25),
        )

    def test_rejection_reason_lists_failed_predicates(self):
        diagnostics = {
            "finite": torch.tensor((True,)),
            "buffer": torch.tensor((False,)),
            "speed": torch.tensor((True,)),
            "curve": torch.tensor((False,)),
        }
        self.assertEqual(rejection_reason(diagnostics, 0), "buffer+curve")

    def test_cross_slot_schedule_is_one_in_four_across_batches(self):
        first = viewer_cross_slots(0, 3, "cpu")
        second = viewer_cross_slots(3, 5, "cpu")
        self.assertEqual(first.tolist(), [True, False, False])
        self.assertEqual(second.tolist(), [False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
