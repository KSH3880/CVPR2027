import tempfile
import unittest
from pathlib import Path

import torch

from trajectory_predictor.checkpoint import load_checkpoint, save_checkpoint
from trajectory_predictor.dataset import sample_states
from trajectory_predictor.geometry import state_to_tokens
from trajectory_predictor.model import JointTrajectoryPredictor


class CheckpointTest(unittest.TestCase):
    def test_save_load_is_bit_exact(self):
        torch.manual_seed(5)
        state, _, _ = sample_states(7, 21)
        model = JointTrajectoryPredictor().eval()
        tokens, _ = state_to_tokens(state)
        model.fit_normalizer(tokens)
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner.pth"
            save_checkpoint(path, model, "a" * 64, epoch=3)
            loaded, payload = load_checkpoint(path)
            with torch.inference_mode():
                after = loaded(state)
        self.assertEqual(payload["epoch"], 3)
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_contract_fails_closed(self):
        model = JointTrajectoryPredictor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner.pth"
            save_checkpoint(path, model, "b" * 64)
            payload = torch.load(path, weights_only=False)
            payload["steer_points"] = 7
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
