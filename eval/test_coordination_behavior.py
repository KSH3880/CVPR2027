import unittest

import numpy as np

from analyze_coordination_behavior import behavior_arrays, audit_geometry, aggregate


class CoordinationBehaviorTests(unittest.TestCase):
    def calculate(self, xy, at, group, half=(.2, .2), yaw=0):
        return behavior_arrays(np.array(xy, dtype=float)[:, None], np.zeros((1, 2)),
            np.array([half]), np.array([yaw]), np.array([at]), np.array([group]))

    def test_strict_pre_at_and_success_delay(self):
        a = self.calculate([(2, 0), (1, 0), (0, 0), (10, 0)], 3, 4)
        self.assertEqual(a['prep_percent'][0], 100)
        self.assertAlmostEqual(a['occupancy_percent'][0], 100/3)
        self.assertEqual(a['pre_frames'][0], 3)
        self.assertAlmostEqual(a['post_at_s'][0], 1/30)
        b = self.calculate([(2, 0), (1, 0), (0, 0)], 2, 2)
        self.assertEqual(b['prep_percent'][0], 50)
        self.assertEqual(b['occupancy_percent'][0], 0)
        self.assertEqual(b['post_at_s'][0], 0)

    def test_initial_success_failures_and_negative_progress(self):
        a = self.calculate([(0, 0), (10, 0)], 1, 1)
        self.assertEqual(a['prep_percent'][0], 0)
        self.assertEqual(a['occupancy_percent'][0], 100)
        a = self.calculate([(1, 0), (2, 0), (3, 0)], 2, -1)
        self.assertEqual(a['prep_percent'][0], -100)
        self.assertTrue(np.isnan(a['post_at_s'][0]))
        a = self.calculate([(1, 0), (0, 0)], -1, -1)
        for key in ('prep_percent','occupancy_percent','post_at_s'):
            self.assertTrue(np.isnan(a[key][0]))
        self.assertEqual(a['pre_frames'][0], 0)

    def test_rotated_rectangle_and_independent_polygon(self):
        xy = [(0, .8), (.8, 0), (0, 0)]
        a = self.calculate(xy, 2, 2, half=(.4, .2), yaw=np.pi/2)
        self.assertEqual(a['occupancy_percent'][0], 50)
        checked = audit_geometry(np.array(xy)[:, None], np.zeros((1, 2)),
            np.array([[.4, .2]]), np.array([np.pi/2]), a)
        self.assertEqual(checked, 1)

    def test_equal_action_weight_and_sample_std(self):
        rows = [dict(seed=seed, metric=value) for seed, values in
                [(1, [0., 10., 20.]), (6, [10., 20., 30.]), (7, [20., 30., 40.])]
                for value in values]
        a = aggregate(rows, ['metric'])['metric']
        self.assertEqual(a['mean'], 20)
        self.assertEqual(a['std'], 10)


if __name__ == '__main__':
    unittest.main()
