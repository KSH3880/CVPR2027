import tempfile
from pathlib import Path
import unittest
import torch

from coordinator.tests.common import make_state
from stack_planner.history import StackHistoryBuffer
from stack_planner.model import StackPlannerConfig
from carry_speed.control import fixed_route
from carry_speed.model import CarrySpeedPolicy


class SpeedPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = CarrySpeedPolicy(StackPlannerConfig(history_steps=2, d_model=32,
                                                         nhead=4, feedforward=64, encoder_layers=1)).eval()
        state = make_state(batch=2)
        self.observation = StackHistoryBuffer(2, 2, "cpu").observe(state, commit=True)
        self.observation.base_path_world = fixed_route(state.root_xy, state.goal_xy)
        self.observation.base_path_valid.fill_(True)
        self.priority = torch.tensor([0, 1])

    def test_leader_action_does_not_affect_ppo_probability_and_route_is_fixed(self):
        path = self.observation.base_path_world.clone()
        action = torch.zeros(2, 2, dtype=torch.long)
        first, _, _ = self.policy.evaluate(self.observation, self.priority, action)
        action[torch.arange(2), self.priority] = 4
        second, entropy, value = self.policy.evaluate(self.observation, self.priority, action)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(path, self.observation.base_path_world))
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool((entropy > 0).all()))
        action[torch.arange(2), 1-self.priority] = 4
        log_prob, _, _ = self.policy.evaluate(self.observation, self.priority, action)
        loss = -log_prob.mean()
        loss.backward()
        self.assertGreater(float(self.policy.actor[-1].weight.grad.abs().sum()), 0.)

    def test_checkpoint_round_trip_preserves_deterministic_control(self):
        expected, _, _ = self.policy.act(self.observation, self.priority, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "speed.pth"
            self.policy.save(file, torch.optim.Adam(self.policy.parameters()), 7, "/tmp/executor.pth", {})
            restored, payload = CarrySpeedPolicy.load(file, "cpu")
            restored.eval()
            actual, _, _ = restored.act(self.observation, self.priority, deterministic=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(payload["step"], 7)


if __name__ == "__main__":
    unittest.main()
