import unittest

import numpy as np
import torch

from coordinator.executor_calibration import (
    fit_first_order_response,
    requested_speeds,
)


class ExecutorCalibrationTest(unittest.TestCase):
    def test_fixed_and_carry_step_profiles(self):
        profiles = torch.arange(6)
        age = torch.full((6, 2), 60)
        speed = requested_speeds(profiles, age)
        self.assertTrue(torch.allclose(speed[0], torch.tensor([0.375, 0.375])))
        self.assertTrue(torch.allclose(speed[3], torch.tensor([1.5, 1.5])))
        self.assertTrue(torch.allclose(speed[4], torch.tensor([1.5, 0.75])))
        self.assertTrue(torch.allclose(speed[5], torch.tensor([1.5, 0.375])))
        age[:, 1] = 130
        restored = requested_speeds(profiles, age)
        self.assertEqual(float(restored[4, 1]), 1.5)
        self.assertEqual(float(restored[5, 1]), 1.5)

    def test_first_order_fit_recovers_simple_response(self):
        dt = 0.1
        alpha, tau = 0.8, 0.5
        sent = np.ones((160, 1, 1), dtype=np.float64)
        sent[:40] = 1.5
        sent[40:100] = 0.75
        sent[100:] = 1.5
        actual = np.zeros_like(sent)
        actual[0] = alpha * sent[0]
        for step in range(sent.shape[0] - 1):
            actual[step + 1] = actual[step] + dt / tau * (
                alpha * sent[step] - actual[step]
            )
        result = fit_first_order_response(
            sent, actual, np.ones_like(sent, dtype=bool), dt
        )
        self.assertAlmostEqual(result["alpha"], alpha, delta=0.05)
        self.assertAlmostEqual(result["tau_s"], tau, delta=0.2)


if __name__ == "__main__":
    unittest.main()
