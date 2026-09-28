import unittest

import torch

from tokenhsi.utils.steer_stop import (
    blend_goal_to_anchor,
    command_speed_reward,
    contract_to_anchor,
    cosine_transition,
    stop_aware_speed_reward,
)


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

    def test_stop_reward_is_continuous_and_maximal_at_physical_stillness(self):
        command = torch.zeros(3)
        progress = torch.tensor([0.0, 1.0e-3, -1.0e-3])
        velocity = torch.tensor([[0.0, 0.0], [1.0e-3, 0.0], [-1.0e-3, 0.0]])
        reward = stop_aware_speed_reward(command, progress, velocity)
        self.assertEqual(reward[0].item(), 1.0)
        self.assertGreater(reward[0].item(), reward[1].item())
        torch.testing.assert_close(reward[1], reward[2])

    def test_move_reward_keeps_the_positive_progress_gate(self):
        command = torch.full((2,), 0.5)
        progress = torch.tensor([0.5, 0.0])
        velocity = torch.zeros(2, 2)
        reward = stop_aware_speed_reward(command, progress, velocity)
        torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))

    def test_dense_overspeed_tail_preserves_exact_tracking_peak(self):
        command = torch.tensor([0.375, 0.375, 0.375])
        progress = torch.tensor([0.375, 1.0, 1.2])
        velocity = torch.stack((progress, torch.zeros_like(progress)), dim=-1)
        reward = command_speed_reward(
            command, progress, velocity,
            overspeed_weight=1.0,
            overspeed_tolerance=0.05,
            overspeed_beta=0.2,
        )
        self.assertEqual(reward[0].item(), 1.0)
        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertGreater(reward[1].item(), reward[2].item())

    def test_direct_stop_penalizes_physical_speed_symmetrically(self):
        command = torch.zeros(3)
        progress = torch.zeros(3)
        velocity = torch.tensor([[0.0, 0.0], [0.8, 0.0], [-0.8, 0.0]])
        reward = command_speed_reward(
            command, progress, velocity,
            overspeed_weight=1.0,
            overspeed_tolerance=0.05,
            overspeed_beta=0.2,
        )
        self.assertEqual(reward[0].item(), 1.0)
        self.assertLess(reward[1].item(), 0.0)
        torch.testing.assert_close(reward[1], reward[2])

    def test_goal_blends_smoothly_to_the_stop_anchor(self):
        final_goal = torch.tensor([[4.0, 2.0, 9.0], [4.0, 2.0, 9.0]])
        anchor = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        height = torch.tensor([0.9, 1.1])
        blend = torch.tensor([0.0, 1.0])
        out = blend_goal_to_anchor(final_goal, anchor, height, blend)
        torch.testing.assert_close(out[0], torch.tensor([4.0, 2.0, 0.9]))
        torch.testing.assert_close(out[1], torch.tensor([1.0, -1.0, 1.1]))


if __name__ == "__main__":
    unittest.main()
