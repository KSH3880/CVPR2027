import unittest

import torch

from tokenhsi.utils.steer_stop import contract_to_anchor, cosine_transition


class SteerStopCommandTest(unittest.TestCase):
    def test_cosine_transition_has_exact_endpoints(self):
        elapsed = torch.tensor([0, 4, 8, 20])
        rising = cosine_transition(elapsed, 8, rising=True)
        falling = cosine_transition(elapsed, 8, rising=False)
        torch.testing.assert_close(rising, torch.tensor([0.0, 0.5, 1.0, 1.0]))
        torch.testing.assert_close(falling, 1.0 - rising)

    def test_hold_keeps_a_fixed_anchor_and_zero_spacing(self):
        points = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        anchor = torch.tensor([[0.6, -0.2]])
        held = contract_to_anchor(points, anchor, torch.ones(1))
        torch.testing.assert_close(held, anchor[:, None, :].expand_as(points))
        torch.testing.assert_close(held[:, 1:] - held[:, :-1], torch.zeros(1, 2, 2))

    def test_brake_changes_only_geometry_without_a_flag_channel(self):
        points = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        anchor = torch.tensor([[0.5, 0.5]])
        out = contract_to_anchor(points, anchor, torch.tensor([0.5]))
        self.assertEqual(out.shape, points.shape)
        torch.testing.assert_close(out, 0.5 * points + 0.5 * anchor[:, None, :])


if __name__ == "__main__":
    unittest.main()
