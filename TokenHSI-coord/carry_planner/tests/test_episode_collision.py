import unittest

import torch

from carry_planner.episode_collision import EpisodeCollisionTracker


class EpisodeCollisionTrackerTest(unittest.TestCase):
    def test_collision_persists_across_steps_and_clears_only_on_done(self):
        tracker = EpisodeCollisionTracker(2, torch.device("cpu"))
        self.assertEqual(tracker.update(torch.tensor([True, False]), torch.tensor([False, False])), (0, 0))
        self.assertEqual(tracker.update(torch.tensor([False, False]), torch.tensor([True, False])), (1, 1))
        self.assertEqual(tracker.update(torch.tensor([False, True]), torch.tensor([True, True])), (1, 2))

    def test_terminal_step_counts(self):
        tracker = EpisodeCollisionTracker(1, torch.device("cpu"))
        self.assertEqual(tracker.update(torch.tensor([True]), torch.tensor([True])), (1, 1))
        self.assertEqual(tracker.update(torch.tensor([False]), torch.tensor([True])), (0, 1))


if __name__ == "__main__":
    unittest.main()
