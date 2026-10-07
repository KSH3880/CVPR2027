import unittest
import torch
from carry_planner.baseline import straight_carry_path


class StraightBaselineTest(unittest.TestCase):
    def test_pickup_and_held_routes(self):
        root = torch.zeros(1, 2, 2)
        box = torch.tensor([[[2., 0.], [2., 0.]]])
        goal = torch.tensor([[[2., 2.], [2., 2.]]])
        path, speed = straight_carry_path(root, box, goal, torch.tensor([[0., 1.]]))
        torch.testing.assert_close(path[..., 0, :], root)
        torch.testing.assert_close(path[..., -1, :], goal)
        torch.testing.assert_close(path[0, 0, 16], box[0, 0])
        torch.testing.assert_close(path[0, 1, 16], goal[0, 1] / 2)
        self.assertEqual(tuple(speed.shape), (1, 2, 33))
        self.assertTrue(bool((speed == 1.5).all()))

    def test_coincident_anchors(self):
        xy = torch.zeros(1, 2, 2)
        path, speed = straight_carry_path(xy, xy, xy, torch.tensor([[0., 1.]]))
        self.assertTrue(bool(torch.isfinite(path).all()))
        self.assertTrue(bool((path == 0).all()))
