import sys
import importlib
import tempfile
import unittest
from pathlib import Path

import torch

from trajectory_predictor.dataset import TrajectoryDataset, generate_split, save_split
from trajectory_predictor.oracle import solve_oracle
from trajectory_predictor.schema import SPEED_VALUES, PlannerState


class OracleDatasetTest(unittest.TestCase):
    def test_package_does_not_import_isaacgym(self):
        importlib.import_module("trajectory_predictor.visualize")
        self.assertNotIn("isaacgym", sys.modules)

    def test_oracle_shapes_speeds_and_clearance(self):
        payload = generate_split(6, 31, batch_size=3)
        speed = payload["speed_class"]
        self.assertEqual(tuple(payload["coarse_path"].shape), (6, 2, 33, 2))
        self.assertTrue(((speed >= 0) & (speed < len(SPEED_VALUES))).all())
        valid = payload["oracle_valid"]
        self.assertTrue((payload["min_clearance"][valid] >= 1.0).all())

    def test_dataset_round_trip(self):
        payload = generate_split(4, 32, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            save_split(path, payload)
            dataset = TrajectoryDataset(path, augment=True)
            sample = dataset[0]
        state = PlannerState.from_mapping({key: sample[key][None] for key in payload["state"]})
        state.validate()
        self.assertEqual(tuple(sample["coarse_path"].shape), (2, 33, 2))


if __name__ == "__main__":
    unittest.main()
