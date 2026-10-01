import unittest
import torch
from carry_speed.control import (action_speeds, fixed_route, generalized_advantages,
                                 rate_limit_speed, rule_speeds, scene_geometry)


class SpeedControlTest(unittest.TestCase):
    def test_zero_reaches_executor_and_acceleration_limits_hold(self):
        speed = torch.tensor([1.5, 0.0])
        desired = torch.tensor([0.0, 1.5])
        for _ in range(60):
            new = rate_limit_speed(speed, desired, 1/30, 0.75, 1.0)
            self.assertTrue(bool(((new-speed) <= 0.75/30+1e-6).all()))
            self.assertTrue(bool(((new-speed) >= -1/30-1e-6).all()))
            speed = new
        self.assertTrue(torch.allclose(speed, desired, atol=1e-5))

    def test_priority_is_independent_of_agent_number(self):
        action = torch.zeros(2, 2, dtype=torch.long)
        speed = action_speeds(action, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(speed, torch.tensor([[1.5, 0.], [0., 1.5]])))

    def test_crossing_and_control_scenes_preserve_routes(self):
        root = torch.tensor([[[-6., 0.], [0., -6.]], [[0., -6.], [-6., 0.]]])
        center = torch.zeros(2, 2)
        start, goal = scene_geometry(root, center, torch.tensor([3., 3.]), torch.zeros(2, 2), torch.tensor([False, True]))
        path = fixed_route(start, goal)
        self.assertTrue(torch.equal(path[..., 0, :], start))
        self.assertTrue(torch.equal(path[..., -1, :], goal))
        for agent in range(2):
            direction = goal[0, agent]-start[0, agent]
            cross = direction[0] * (-start[0, agent, 1]) - direction[1] * (-start[0, agent, 0])
            self.assertEqual(float(cross), 0.)
        self.assertEqual(float((start[1, 0]-start[1, 1]).norm()), 3.)
        self.assertEqual(float((goal[1, 0]-goal[1, 1]).norm()), 3.)

    def test_reference_yields_then_resumes_and_free_scene_keeps_speed(self):
        center = torch.zeros(2, 2)
        priority = torch.tensor([0, 1])
        root = torch.tensor([[[-2., 0.], [0., -2.]], [[0., -2.], [-2., 0.]]])
        goal = -root
        speed = rule_speeds(root, center, goal, priority)
        self.assertTrue(torch.equal(speed, torch.tensor([[1.5, 0.], [0., 1.5]])))
        root[0, 0, 0] = 1.8
        root[1, 1, 0] = 1.8
        self.assertTrue(torch.equal(rule_speeds(root, center, goal, priority), torch.full((2, 2), 1.5)))

    def test_gae_does_not_leak_across_reset(self):
        reward = torch.tensor([[1.], [2.], [100.]])
        done = torch.tensor([[False], [True], [False]])
        value = torch.zeros_like(reward)
        advantage, returns = generalized_advantages(reward, done, value, value, gamma=1., lam=1.)
        self.assertTrue(torch.equal(advantage, torch.tensor([[3.], [2.], [100.]])))
        self.assertTrue(torch.equal(returns, advantage))


if __name__ == "__main__":
    unittest.main()
