#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import autofill
import eval_runner
import queue_status
import queue_results
import slot_runner
from queue_docs import read_queue_rows, replace_block, update_queue


QUEUE = """# 사람 영역

<!-- QUEUE -->

| tag | 질문·판정 기준 | 상태/GPU | 환경변수 |
|---|---|---|---|
| `a` | first |  | `A=1` |
| `b` | second | gpu1 | `A=2` |

<!-- /QUEUE -->

## 사람 메모
keep me
"""


class QueueDocumentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "PLAN_test.md"
        self.path.write_text(QUEUE)

    def tearDown(self):
        self.temp.cleanup()

    def test_status_update_preserves_human_sections(self):
        update_queue(self.path, statuses={"a": "gpu2 · 실행 중", "b": "대기"})
        text = self.path.read_text()
        self.assertIn("# 사람 영역", text)
        self.assertIn("## 사람 메모\nkeep me", text)
        self.assertIn("| `a` | first | gpu2 · 실행 중 | `A=1` |", text)
        self.assertIn("\n<!-- /QUEUE -->", text)
        self.assertEqual([row["tag"] for row in read_queue_rows(self.path)], ["a", "b"])

    def test_remove_only_requested_terminal_row(self):
        removed = update_queue(self.path, remove_tags={"a"})
        self.assertEqual([row["tag"] for row in removed], ["a"])
        self.assertEqual(removed[0]["note"], "first")
        self.assertEqual(removed[0]["env"], "A=1")
        self.assertEqual([row["tag"] for row in read_queue_rows(self.path)], ["b"])
        self.assertIn("keep me", self.path.read_text())

    def test_generated_block_does_not_touch_neighbors(self):
        dashboard = Path(self.temp.name) / "PLAN.md"
        dashboard.write_text("before\n<!-- GPUMAP -->old<!-- /GPUMAP -->\nafter\n")
        replace_block(dashboard, "<!-- GPUMAP -->", "<!-- /GPUMAP -->",
                      "<!-- GPUMAP -->new<!-- /GPUMAP -->")
        self.assertEqual(dashboard.read_text(),
                         "before\n<!-- GPUMAP -->new<!-- /GPUMAP -->\nafter\n")

    def test_master_queue_summary_mirrors_track_rows(self):
        track = {"name": "test", "queue": self.path}
        block = queue_status.queue_summary_block(
            [track], {"test": {"a": "gpu2 · 실행 중", "b": "대기"}})
        self.assertIn("| [test](PLAN_test.md) | `a` | first | gpu2 · 실행 중 | `A=1` |", block)
        self.assertIn("| [test](PLAN_test.md) | `b` | second | 대기 | `A=2` |", block)

    def test_running_job_missing_from_track_queue_needs_attention(self):
        state = Path(self.temp.name) / "state.json"
        state.write_text('{"jobs": {}}\n')
        track = {"name": "test", "queue": self.path, "state": state}
        jobs = {0: [{"track": "test", "tag": "outside", "eval": False,
                    "reserved": False}]}
        self.assertIn(("test", "outside", "실행 중이지만 트랙 활성 큐에 없음"),
                      queue_status.attention_rows([track], jobs))

    def test_unsafe_env_is_rejected(self):
        track = {"queue": self.path, "allowed_env": ["A"], "forbidden_env": []}
        with self.assertRaises(ValueError):
            autofill.parse_env(track, "bad", "A=$(touch_bad)")

    def test_unsafe_tag_is_rejected(self):
        self.path.write_text(QUEUE.replace("`a`", "`a;bad`"))
        track = {"name": "test", "queue": self.path,
                 "state": Path(self.temp.name) / "state.json",
                 "allowed_env": ["A"], "forbidden_env": [], "max_seeds": 2}
        with self.assertRaises(ValueError):
            autofill.read_queue(track)

    def test_redundant_base_env_is_rejected(self):
        track = {"name": "test", "queue": self.path,
                 "state": Path(self.temp.name) / "state.json", "base_env": "A=1",
                 "allowed_env": ["A"], "forbidden_env": [], "max_seeds": 2}
        with self.assertRaises(ValueError):
            autofill.read_queue(track)

    def test_shared_gpu_prefers_lower_memory_and_respects_headroom(self):
        gpu_rows = "\n".join([
            "0, u0, 43000, 80000",
            "1, u1, 18000, 80000",
            "2, u2, 60000, 80000",
            "3, u3, 1000, 80000",
            "4, u4, 18000, 80000",
        ])
        apps = "\n".join([
            "u0, 100", "u1, 101", "u2, 102", "u4, 103", "u4, 104",
        ])
        responses = [mock.Mock(stdout=gpu_rows), mock.Mock(stdout=apps)]
        with mock.patch.object(autofill.subprocess, "run", side_effect=responses), \
             mock.patch.object(autofill, "slot_load", return_value={}):
            idle, shared = autofill.gpu_slots(2)
        self.assertEqual(idle, [3])
        self.assertEqual(shared, [1, 0])

    def test_terminal_is_archived_before_queue_removal(self):
        state_path = Path(self.temp.name) / "state.json"
        auto_path = Path(self.temp.name) / "AUTO.md"
        state_path.write_text('{"started": ["a"]}\n')
        track = {"name": "steer", "queue": self.path, "state": state_path,
                 "auto_results": auto_path, "eval": "eval.sh", "base_env": "",
                 "allowed_env": ["A"], "forbidden_env": []}

        def refresh(_):
            self.assertIn("a", [row["tag"] for row in read_queue_rows(self.path)])
            auto_path.write_text("saved\n")

        with mock.patch.object(autofill, "result_tags", return_value={"a"}), \
             mock.patch.object(autofill, "refresh_results", side_effect=refresh), \
             mock.patch.object(autofill.subprocess, "run", return_value=mock.Mock(stdout="")), \
             mock.patch.object(autofill.subprocess, "Popen"):
            self.assertTrue(autofill.consume_terminal_rows(track, {"started": ["a"]}))
        self.assertTrue(auto_path.exists())
        self.assertNotIn("a", [row["tag"] for row in read_queue_rows(self.path)])
        self.assertIn('"status": "완료"', state_path.read_text())

    def test_queue_row_survives_auto_result_failure(self):
        state_path = Path(self.temp.name) / "state.json"
        state_path.write_text('{"started": ["a"]}\n')
        track = {"name": "steer", "queue": self.path, "state": state_path,
                 "auto_results": Path(self.temp.name) / "AUTO.md", "eval": "eval.sh",
                 "base_env": "", "allowed_env": ["A"], "forbidden_env": []}
        with mock.patch.object(autofill, "result_tags", return_value={"a"}), \
             mock.patch.object(autofill, "refresh_results", side_effect=OSError("disk full")):
            self.assertFalse(autofill.consume_terminal_rows(track, {"started": ["a"]}))
        self.assertIn("a", [row["tag"] for row in read_queue_rows(self.path)])

    def test_result_append_is_idempotent_and_rejects_conflict(self):
        results = Path(self.temp.name) / "results.tsv"
        fields = ["run_a", "0.9", "0.1", "0.8", "0.2", "0.7", "1.0", "-"]
        with mock.patch.object(queue_results, "RESULTS", results), \
             mock.patch.object(queue_results, "refresh"):
            self.assertTrue(queue_results.append(fields))
            self.assertFalse(queue_results.append(fields))
            with self.assertRaises(ValueError):
                queue_results.append(["run_a", "0.8", "0.1", "0.8", "0.2", "0.7", "1.0", "-"])
        self.assertEqual(results.read_text().count("run_a"), 1)

    def test_curated_auto_results_are_newest_first(self):
        auto = Path(self.temp.name) / "AUTO.md"
        curated = Path(self.temp.name) / "EXPERIMENTS.md"
        curated.write_text("# results\n\n<!-- AUTO_RESULTS -->old<!-- /AUTO_RESULTS -->\n\nmanual\n")
        track = {"name": "ma", "auto_results": auto, "curated_results": str(curated)}
        state = {"jobs": {
            "old": {"status": "완료", "finished_at": "2026-08-17T10:00:00+09:00"},
            "new": {"status": "완료", "finished_at": "2026-08-18T10:00:00+09:00"},
            "unknown": {"status": "완료"},
        }}
        with mock.patch.object(queue_results, "ROOT", Path("/")), \
             mock.patch.object(queue_results, "load_track", return_value=track), \
             mock.patch.object(queue_results, "load_state", return_value=state):
            queue_results.refresh("ma")
        text = curated.read_text()
        self.assertLess(text.index("`new`"), text.index("`old`"))
        self.assertLess(text.index("`old`"), text.index("`unknown`"))
        self.assertIn("manual", text)

    def test_slot_runner_records_exit_before_unlock(self):
        root = Path(self.temp.name)
        logs = root / "runs/queue/logs"
        logs.mkdir(parents=True)
        (logs / "job.env.json").write_text('{"A": "1"}\n')
        lock = root / "gpu0.lock"
        lock.write_text("1\n")
        exit_record = root / "exit.json"
        completed = mock.Mock(returncode=0)
        track = {"name": "test", "train": "train.sh", "tag_env": "TEST_TAG"}
        with mock.patch.object(slot_runner, "ROOT", root), \
             mock.patch.object(slot_runner, "load_track", return_value=track), \
             mock.patch.object(slot_runner, "exit_path", return_value=exit_record), \
             mock.patch.object(slot_runner.subprocess, "run", return_value=completed):
            self.assertEqual(slot_runner.main("test", "job", "0", str(lock)), 0)
        self.assertFalse(lock.exists())
        self.assertIn('"rc": 0', exit_record.read_text())

    def test_eval_modes_only_gate_training_rows(self):
        track = {"name": "ma", "queue": self.path, "eval": "eval.sh",
                 "eval_mode_env": "MA_MODE", "eval_modes": ["train", "adapt"],
                 "base_env": "", "allowed_env": ["MA_MODE", "QUEUE_EVAL"],
                 "forbidden_env": []}
        self.assertTrue(autofill.requires_eval(
            track, {"tag": "train", "env": "MA_MODE=adapt"}))
        self.assertFalse(autofill.requires_eval(
            track, {"tag": "probe", "env": "MA_MODE=sweep"}))
        self.assertTrue(autofill.requires_eval(
            track, {"tag": "forced", "env": "MA_MODE=sweep QUEUE_EVAL=1"}))

    def test_training_row_waits_for_required_eval(self):
        track = {"name": "ma", "queue": self.path, "eval": "eval.sh",
                 "eval_mode_env": "MA_MODE", "eval_modes": ["adapt"],
                 "base_env": "", "allowed_env": ["MA_MODE"], "forbidden_env": []}
        row = {"tag": "a", "env": "MA_MODE=adapt", "note": "", "status": ""}
        exit_record = {"status": "training_complete", "rc": 0}
        with mock.patch.object(autofill, "eval_summary", return_value=""), \
             mock.patch.object(autofill, "load_exit", return_value=exit_record):
            self.assertIsNone(autofill._terminal(track, row, {"a"}, ""))
        with mock.patch.object(autofill, "eval_summary",
                               return_value="MA_EVAL_SUMMARY tag=a rc=0"):
            result = autofill._terminal(track, row, {"a"}, "")
        self.assertEqual(result["status"], "완료")

    def test_eval_failure_is_recorded_before_unlock(self):
        root = Path(self.temp.name)
        (root / "runs/queue/logs").mkdir(parents=True)
        lock = root / "gpu0.lock"
        lock.write_text("1\n")
        responses = [mock.Mock(returncode=1), mock.Mock(returncode=0)]
        with mock.patch.object(eval_runner, "ROOT", root), \
             mock.patch.object(eval_runner, "load_track",
                               return_value={"eval": "eval.sh"}), \
             mock.patch.object(eval_runner.subprocess, "run", side_effect=responses):
            self.assertEqual(eval_runner.main(
                "ma", "job", "0", "job", "/tmp/checkpoint", str(lock)), 1)
        self.assertFalse(lock.exists())
        failure = root / "runs/queue/eval_failures/ma/job.json"
        self.assertIn('"rc": 1', failure.read_text())


if __name__ == "__main__":
    unittest.main()
