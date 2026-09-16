import unittest
from stack_planner.retreat_metrics import summarize_retreat


class RetreatMetricsTest(unittest.TestCase):
    def test_empty_rates_are_not_zero_success(self):
        result = summarize_retreat([])
        self.assertIsNone(result['retreat_reach_ratio'])
        self.assertEqual(result['retreat_episodes'], 0)

    def test_censored_episode_excluded_from_rates_not_frames(self):
        row = dict(completed=True, retreat_steps=2, reached=True, stopped=False,
                   fall=False, endpoint_changes=[.2], path_errors=[.1, .3],
                   collision_steps=1, min_box_gap=.4, min_agent_gap=.5,
                   approaching_steps=1, stationary_steps=0)
        censored = dict(row, completed=False, reached=False, path_errors=[.6])
        result = summarize_retreat([row, censored])
        self.assertEqual(result['retreat_reach_ratio'], 1.)
        self.assertEqual(result['censored_episodes'], 1)
        self.assertAlmostEqual(result['retreat_path_mae'], 1./3.)
        self.assertEqual(result['collision_proxy_ratio'], .5)
        self.assertEqual(result['approaching_episodes'], 1)
        self.assertIsNone(result['stationary_reach_ratio'])
