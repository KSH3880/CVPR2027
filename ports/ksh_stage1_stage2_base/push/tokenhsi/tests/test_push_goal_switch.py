import unittest
import torch
from utils.push_goal_switch import rotated_goal


class GoalSwitchTests(unittest.TestCase):
    def test_turn_left_and_reverse_from_current_box_keep_height(self):
        box = torch.tensor([[[3., 4., .65], [-2., 6., .65]]])
        heading = torch.tensor([[[1., 0.], [0., 1.]]])
        old = torch.tensor([[[9., 8., .65], [4., 8., .8]]])
        left = rotated_goal(box, heading, old, 90, 2)
        reverse = rotated_goal(box, heading, old, 180, 2)
        torch.testing.assert_close(left, torch.tensor([[[3., 6., .65], [-4., 6., .8]]]))
        torch.testing.assert_close(reverse, torch.tensor([[[1., 4., .65], [-2., 4., .8]]]))
        torch.testing.assert_close(box, torch.tensor([[[3., 4., .65], [-2., 6., .65]]]))
        torch.testing.assert_close(old, torch.tensor([[[9., 8., .65], [4., 8., .8]]]))


if __name__ == '__main__':
    unittest.main()
