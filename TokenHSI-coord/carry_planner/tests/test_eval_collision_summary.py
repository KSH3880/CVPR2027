import unittest

from carry_planner.eval_collision_summary import summarize


class EvalCollisionSummaryTest(unittest.TestCase):
    def payload(self):
        return dict(definition="body center distance", threshold_m=.3, records=[
            dict(repeat=r, env=e, agent=a, executed_steps=100,
                 collision_steps=10 if r == 0 or (r == 1 and e == 0) else 0,
                 success=e == 1)
            for r in range(3) for e in range(2) for a in range(2)])

    def test_excludes_warmup_and_counts_each_environment_once(self):
        result = summarize(self.payload())
        self.assertEqual(result["completed_episodes"], 4)
        self.assertEqual(result["collided_episodes"], 1)
        self.assertEqual(result["collision_episode_pair"], .25)
        self.assertEqual(result["collision_step_fraction"], .025)

    def test_rejects_duplicate_and_incomplete_collection(self):
        data = self.payload()
        data["records"].append(data["records"][0])
        with self.assertRaises(ValueError):
            summarize(data)
        data = self.payload()
        data["records"].pop()
        with self.assertRaises(ValueError):
            summarize(data)
