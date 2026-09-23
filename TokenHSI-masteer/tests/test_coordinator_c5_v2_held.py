import unittest

import torch

from coordinator.schema import CoordinatorState
from coordinator_c5_v2 import C5TrajectoryCodec


class C5V2HeldCodecTest(unittest.TestCase):
    def test_held_agent_ignores_completed_approach_knots(self):
        root = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]])
        state = CoordinatorState(
            root_xy=root,
            heading=torch.zeros(1, 2),
            root_vel_xy=torch.zeros(1, 2, 2),
            box_xyz=torch.cat((root, torch.full((1, 2, 1), 0.8)), dim=-1),
            box_heading=torch.zeros(1, 2),
            box_vel_xy=torch.zeros(1, 2, 2),
            box_size_xy=torch.full((1, 2, 2), 0.5),
            goal_xy=torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
            held=torch.ones(1, 2),
            phase=torch.full((1, 2), 2.0),
        )
        control = torch.zeros(1, 1, 8, 2, 3)
        control[:, :, :4, :, :2] = 2.0
        control[:, :, 4:, :, 1] = 0.5
        path = C5TrajectoryCodec()(control.reshape(1, 1, 48), state)["path_world"]
        torch.testing.assert_close(
            path[0, 0, :, :17], root[0, :, None].expand(-1, 17, -1)
        )
        self.assertGreater(
            (path[0, 0, :, 17:] - state.goal_xy[0, :, None]).abs().sum().item(),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
