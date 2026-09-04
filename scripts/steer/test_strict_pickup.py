import unittest

import numpy as np

from strict_pickup import episode_records, max_true_run


class FakeNpz(dict):
    @property
    def files(self):
        return list(self)


class StrictPickupTest(unittest.TestCase):
    def test_consecutive_steps(self):
        self.assertEqual(max_true_run([1, 1, 0, 1, 1, 1]), 3)

    def test_requires_close_lift_and_streak(self):
        steps = 14
        data = FakeNpz(
            box=np.zeros((steps, 1, 3), np.float32),
            episode=np.ones((steps, 1), np.int32),
            held=np.ones((steps, 1), np.uint8),
            ep0=np.array([2]),
            ep1=np.array([1]),
            z00=np.array([0.5]),
            z01=np.array([0.5]),
        )
        data["box"][:, 0, 2] = 0.61
        data["held"][6, 0] = 0

        record = episode_records(data, min_steps=1)[0]
        self.assertFalse(record["pickup"])
        self.assertEqual(record["max_streak"], 7)

        data["held"][6, 0] = 1
        record = episode_records(data, min_steps=1)[0]
        self.assertTrue(record["pickup"])
        self.assertEqual(record["max_streak"], 14)


if __name__ == "__main__":
    unittest.main()
