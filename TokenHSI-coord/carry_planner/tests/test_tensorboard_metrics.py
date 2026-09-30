import json
import tempfile
import unittest
from pathlib import Path

from carry_planner.tensorboard_metrics import (
    import_jsonl, make_writer, write_scalars,
)


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.flushes = 0

    def add_scalar(self, tag, value, global_step):
        self.scalars.append((tag, value, global_step))

    def flush(self):
        self.flushes += 1


class CarryTensorBoardMetricsTest(unittest.TestCase):
    def test_selected_finite_metrics_keep_iteration(self):
        writer = RecordingWriter()
        count = write_scalars(writer, {
            "iteration": 17,
            "reward": 0.25,
            "collision_ratio": 0.1,
            "collision_episode_fraction": 0.8,
            "analytic_collision_loss": float("nan"),
            "unrelated": 123.0,
        })
        self.assertEqual(count, 3)
        self.assertEqual(writer.scalars, [
            ("Reward/total", 0.25, 17),
            ("Collision/proxy_step_fraction", 0.1, 17),
            ("Collision/proxy_episode_fraction", 0.8, 17),
        ])

    def test_import_ignores_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(
                json.dumps({"iteration": 1, "reward": 0.2}) + "\n"
                + '{"iteration": 2, "reward":', encoding="utf-8",
            )
            writer = RecordingWriter()
            self.assertEqual(import_jsonl(metrics, writer), 1)
            self.assertEqual(writer.flushes, 1)
            self.assertEqual(writer.scalars, [("Reward/total", 0.2, 1)])

    def test_real_event_file_contains_scalars(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("tensorboard is not installed")
        with tempfile.TemporaryDirectory() as directory:
            writer = make_writer(Path(directory))
            write_scalars(writer, {
                "iteration": 5, "reward": 0.3,
                "sample_plan_valid_fraction": 0.7,
            })
            writer.close()
            events = EventAccumulator(directory)
            events.Reload()
            self.assertEqual(
                events.Scalars("Reward/total")[0].step, 5,
            )
            self.assertAlmostEqual(
                events.Scalars("Plan/sample_valid_fraction")[0].value, 0.7,
            )


if __name__ == "__main__":
    unittest.main()
