import unittest

import torch

from coordinator.schema import MAX_SPEED, MIN_SPEED, CoordinatorState
from coordinator_c5_v2 import (
    ACTION_DIM,
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
    MANEUVER_STRAIGHT,
    C5Config,
    C5ProposalActor,
    C5TrajectoryCodec,
    canonical_action,
    encode_state,
)


def make_state(batch=4):
    root = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]]).expand(batch, -1, -1).clone()
    box_xy = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]]).expand(batch, -1, -1).clone()
    goal = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]).expand(batch, -1, -1).clone()
    return CoordinatorState(
        root_xy=root, heading=torch.zeros(batch, 2), root_vel_xy=torch.zeros(batch, 2, 2),
        box_xyz=torch.cat((box_xy, torch.full((batch, 2, 1), 0.35)), dim=-1),
        box_heading=torch.zeros(batch, 2), box_vel_xy=torch.zeros(batch, 2, 2),
        box_size_xy=torch.full((batch, 2, 2), 0.5), goal_xy=goal,
        held=torch.zeros(batch, 2), phase=torch.zeros(batch, 2),
    )


class C5V2CodecTest(unittest.TestCase):
    def test_48d_contract_anchors_and_speed_bounds(self):
        state = make_state()
        role = torch.tensor([0, 0, 1, 1])
        maneuver = torch.tensor([
            MANEUVER_STRAIGHT, MANEUVER_LEFT, MANEUVER_RIGHT, MANEUVER_SLOW
        ])
        action = canonical_action(role, maneuver)
        self.assertEqual(action.shape, (4, ACTION_DIM))
        decoded = C5TrajectoryCodec()(action[:, None], state)
        self.assertEqual(decoded["path_world"].shape, (4, 1, 2, 33, 2))
        torch.testing.assert_close(decoded["path_world"][:, 0, :, 0], state.root_xy)
        torch.testing.assert_close(decoded["path_world"][:, 0, :, 16], state.box_xyz[..., :2])
        torch.testing.assert_close(decoded["path_world"][:, 0, :, 32], state.goal_xy)
        self.assertTrue((decoded["speed"] >= MIN_SPEED).all())
        self.assertTrue((decoded["speed"] <= MAX_SPEED).all())

    def test_canonical_left_right_and_slowdown(self):
        state = make_state()
        role = torch.zeros(4, dtype=torch.long)
        maneuver = torch.tensor([
            MANEUVER_STRAIGHT, MANEUVER_LEFT, MANEUVER_RIGHT, MANEUVER_SLOW
        ])
        decoded = C5TrajectoryCodec()(canonical_action(role, maneuver)[:, None], state)
        path = decoded["path_world"][:, 0]
        left_offset = path[1, 1, :, 1] - path[0, 1, :, 1]
        right_offset = path[2, 1, :, 1] - path[0, 1, :, 1]
        self.assertGreater(left_offset.max().item(), 0.1)
        self.assertLess(right_offset.min().item(), -0.1)
        slow_speed = decoded["speed"][3, 0, 1].mean()
        straight_speed = decoded["speed"][0, 0, 1].mean()
        self.assertLess(slow_speed.item(), straight_speed.item())
        self.assertGreaterEqual(slow_speed.item(), MIN_SPEED)

    def test_role_and_maneuver_conditioned_actor_contract(self):
        state = make_state(batch=2)
        config = C5Config(history_frames=4, hidden_dim=32)
        actor = C5ProposalActor(config)
        history = encode_state(state)[:, None].expand(-1, 4, -1).clone()
        action = actor.sample(
            history, torch.tensor([0, 1]), torch.tensor([MANEUVER_LEFT, MANEUVER_RIGHT]),
            candidates=5, generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(action.shape, (2, 5, ACTION_DIM))


if __name__ == "__main__":
    unittest.main()
