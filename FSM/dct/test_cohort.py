import unittest

from cohort import OUT, SETTINGS, SEEDS, MODELS, CONDITION, env_count, schedule, jobs, command


class CohortTests(unittest.TestCase):
    def test_balanced_coverage(self):
        for seed in SEEDS:
            for family in ('enable', 'coupled'):
                self.assertEqual(sum(env_count(s, seed) for s in SETTINGS if s.startswith(family)), 512)
        for setting in SETTINGS:
            self.assertEqual(sum(env_count(setting, seed) for seed in SEEDS), 512)
        self.assertEqual(len(jobs()), 180)
        self.assertEqual(sum(env_count(j['setting'], j['seed']) for j in jobs()), 30720)

    def test_schedule_and_command_pairing(self):
        for seed in SEEDS:
            for setting in SETTINGS:
                random = schedule(setting, seed, CONDITION)
                self.assertEqual(random, schedule(setting, seed, CONDITION))
                self.assertEqual(len(random['steps']), env_count(setting, seed))
                self.assertTrue(all(30 <= x <= 150 for x in random['steps']))
                self.assertTrue(all(x == 0 for x in schedule(setting, seed, 0)['steps']))
                paths = set()
                for model in MODELS:
                    j = dict(setting=setting, model=model, seed=seed, delay=CONDITION)
                    cmd, env = command(j, OUT/'unused', gpu=6)
                    zero, ze = command(dict(j, delay=0), OUT/'unused', gpu=6)
                    self.assertEqual(cmd, zero)
                    self.assertEqual({k:v for k,v in env.items() if k != 'DCT_DELAY_SCHEDULE'},
                                     {k:v for k,v in ze.items() if k != 'DCT_DELAY_SCHEDULE'})
                    paths.add(env['DCT_DELAY_SCHEDULE'])
                self.assertEqual(len(paths), 1)


if __name__ == '__main__':
    unittest.main()
