from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from coordinator.checkpoint import load_checkpoint, save_checkpoint
from coordinator.model import JointCoordinator
from coordinator.tests.common import make_state


class CheckpointTest(unittest.TestCase):
    def test_round_trip_is_bit_exact(self):
        torch.manual_seed(7)
        state = make_state(1)
        model = JointCoordinator().eval()
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coord.pth"
            save_checkpoint(path, model, step=3)
            loaded, payload = load_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["step"], 3)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)


if __name__ == "__main__":
    unittest.main()
