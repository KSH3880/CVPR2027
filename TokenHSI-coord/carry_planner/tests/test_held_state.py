import unittest

import torch

from carry_planner.held_state import observed_box_held


class ObservedBoxHeldTest(unittest.TestCase):
    def setUp(self):
        self.root = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        self.box = torch.tensor([[[2.0, 0.0, 0.8], [0.2, 0.0, 0.8]]])
        self.hands = self.box.clone()
        self.size_z = torch.full((1, 2), 0.4)

    def test_high_box_far_from_agent_is_not_held(self):
        held = observed_box_held(
            self.root, self.box, self.size_z, self.hands,
        )
        self.assertEqual(held.tolist(), [[False, True]])

    def test_hand_must_be_near_box(self):
        hands = self.hands.clone()
        hands[:, 1, 0] += 0.5
        held = observed_box_held(
            self.root, self.box, self.size_z, hands,
        )
        self.assertEqual(held.tolist(), [[False, False]])

    def test_box_must_be_lifted(self):
        box = self.box.clone()
        box[:, 1, 2] = 0.25
        held = observed_box_held(
            self.root, box, self.size_z, box,
        )
        self.assertEqual(held.tolist(), [[False, False]])


if __name__ == "__main__":
    unittest.main()
