import unittest

import numpy as np
import torch

from carry_planner.view_debug import (
    cap_box_approach_distance, rejected_path_vertices, rejection_reason,
    viewer_cross_slots, viewer_walk_box_shift,
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

    def test_box_approach_cap_preserves_near_and_direction(self):
        root = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]]])
        box = torch.tensor([[[6.0, 0.0], [3.0, 2.0], [1.0, 1.0]]])
        adjusted = cap_box_approach_distance(root, box, 3.0)
        self.assertTrue(torch.allclose(adjusted[0, 0], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.equal(adjusted[0, 1:], box[0, 1:]))
        with self.assertRaises(ValueError):
            cap_box_approach_distance(root, box, 0.0)

    def test_walk_box_shift_handles_missing_reset_and_agent_rows(self):
        roots = torch.zeros(4, 13)
        roots[1, :2] = torch.tensor([2.0, 0.0])
        roots[2, :2] = torch.tensor([10.0, 0.0])
        boxes = torch.zeros(2, 2, 13)
        boxes[0, 1, :2] = torch.tensor([8.0, 0.0])
        boxes[1, 0, :2] = torch.tensor([12.0, 0.0])
        self.assertIsNone(viewer_walk_box_shift(None, roots, boxes, 2, 3.0))
        rows = {"carry": {"loco_carry": torch.tensor([1, 2])}}
        env, agent, shift = viewer_walk_box_shift(rows, roots, boxes, 2, 3.0)
        self.assertEqual(env.tolist(), [0, 1])
        self.assertEqual(agent.tolist(), [1, 0])
        self.assertTrue(torch.allclose(shift, torch.tensor([[-3.0, 0.0], [0.0, 0.0]])))

    def test_cross_slot_schedule_is_one_in_four_across_batches(self):
        first = viewer_cross_slots(0, 3, "cpu")
        second = viewer_cross_slots(3, 5, "cpu")
        self.assertEqual(first.tolist(), [True, False, False])
        self.assertEqual(second.tolist(), [False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
