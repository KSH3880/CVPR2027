import unittest

import torch

from carry_planner.episode import initial_timeout_deadlines


class InitialTimeoutDeadlinesTest(unittest.TestCase):
    def test_deadlines_cover_episode_without_zero_length(self):
        torch.manual_seed(7)
        deadlines = initial_timeout_deadlines(2048, 600, torch.device("cpu"))
        self.assertEqual(deadlines.shape, (2048,))
        self.assertTrue(bool((deadlines >= 3).all()))
        self.assertTrue(bool((deadlines <= 600).all()))
        self.assertGreater(int(deadlines.unique().numel()), 400)

    def test_invalid_episode_length(self):
        with self.assertRaises(ValueError):
            initial_timeout_deadlines(8, 2, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
