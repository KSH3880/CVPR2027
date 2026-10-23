import unittest
import torch
from utils.push_box_switch import relocation_ray


class BoxSwitchTests(unittest.TestCase):
    def test_across_goal_crosses_goal_instead_of_rotating_goal(self):
        boxes = torch.tensor([[[1., 2., .65]]])
        goals = torch.tensor([[[4., 2., .65]]])
        roots = torch.zeros_like(boxes)
        q = torch.tensor([[[0., 0., 0., 1.]]])
        anchor, direction = relocation_ray(boxes, goals, roots, q,
            torch.tensor([[[1., 0.]]]), 'across_goal')
        torch.testing.assert_close(anchor+2*direction, torch.tensor([[[6., 2.]]]))
        torch.testing.assert_close(goals, torch.tensor([[[4., 2., .65]]]))

    def test_behind_agent_tracks_current_facing_not_goal(self):
        boxes = torch.tensor([[[1., 2., .65]]])
        goals = torch.tensor([[[4., 2., .65]]])
        roots = torch.tensor([[[3., 7., 1.]]])
        s = 2**-.5
        q = torch.tensor([[[0., 0., s, s]]])
        anchor, direction = relocation_ray(boxes, goals, roots, q,
            torch.tensor([[[1., 0.]]]), 'behind_agent')
        torch.testing.assert_close(anchor+2*direction, torch.tensor([[[3., 5.]]]))

    def test_goal_center_and_vertical_facing_have_finite_fallbacks(self):
        xyz = torch.zeros(1, 1, 3)
        s = 2**-.5
        q = torch.tensor([[[0., s, 0., s]]])
        initial = torch.tensor([[[0., 1.]]])
        for mode, expected in [('across_goal', initial),
                               ('behind_agent', torch.tensor([[[-1., 0.]]]))]:
            anchor, direction = relocation_ray(xyz, xyz, xyz, q, initial, mode)
            self.assertTrue(torch.isfinite(direction).all())
            torch.testing.assert_close(direction, expected)


if __name__ == '__main__':
    unittest.main()
