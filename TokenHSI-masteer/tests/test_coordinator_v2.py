import tempfile
import unittest
from pathlib import Path

import torch

from coordinator.schema import CoordinatorState
from coordinator_v2 import PlannerConfig, WorldModelPlanner, encode_state
from coordinator_v2.checkpoint import load_checkpoint, save_checkpoint


def make_state(batch=2):
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


class CoordinatorV2Test(unittest.TestCase):
    def setUp(self):
        self.cfg = PlannerConfig(history_frames=4, plan_horizon=4, hidden_dim=32,
                                 ensemble_size=2, mppi_candidates=8,
                                 mppi_iterations=2, result_candidates=3)

    def test_plan_contract_and_anchors(self):
        state = make_state()
        planner = WorldModelPlanner(self.cfg).eval()
        history = encode_state(state)[:, None].expand(-1, self.cfg.history_frames, -1).clone()
        result = planner.plan(history, state, generator=torch.Generator().manual_seed(7))
        self.assertEqual(result.path_world.shape, (2, 2, 33, 2))
        self.assertEqual(result.speed.shape, (2, 2, 33))
        self.assertEqual(result.candidate_path_world.shape, (2, 3, 2, 33, 2))
        torch.testing.assert_close(result.path_world[:, :, 0], state.root_xy)
        torch.testing.assert_close(result.path_world[:, :, 16], state.box_xyz[..., :2])
        torch.testing.assert_close(result.path_world[:, :, 32], state.goal_xy)
        self.assertTrue(torch.isfinite(result.path_world).all())

    def test_alignment_and_checkpoint(self):
        state = make_state()
        planner = WorldModelPlanner(self.cfg)
        history = encode_state(state)[:, None].expand(-1, self.cfg.history_frames, -1).clone()
        context = planner.encode_history(history)
        action = torch.zeros(2, self.cfg.plan_horizon, 2, 3)
        dwell = torch.zeros(2, 2)
        loss = planner.proposal.alignment_loss(context, action, dwell, torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.pth"
            save_checkpoint(path, planner, step=3)
            restored, payload = load_checkpoint(path)
            self.assertEqual(payload["step"], 3)
            self.assertEqual(restored.config, planner.config)


if __name__ == "__main__":
    unittest.main()
