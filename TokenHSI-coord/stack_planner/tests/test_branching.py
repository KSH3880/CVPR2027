from __future__ import annotations

import unittest

import torch

from stack_planner.branching import TaskBranchSnapshot


class TaskBranchSnapshotTest(unittest.TestCase):
    def test_update_where_merges_whole_env_for_factored_buffers(self):
        target = TaskBranchSnapshot(
            tensors={
                "_root_states": torch.zeros(6, 2),
                "_dof_state": torch.zeros(12, 2),
            },
            factors={"_root_states": 2, "_dof_state": 4},
            num_envs=3,
        )
        source = TaskBranchSnapshot(
            tensors={
                "_root_states": torch.arange(12.0).reshape(6, 2),
                "_dof_state": torch.arange(24.0).reshape(12, 2),
            },
            factors={"_root_states": 2, "_dof_state": 4},
            num_envs=3,
        )
        target.update_where(source, torch.tensor([False, True, False]))
        self.assertTrue(torch.equal(
            target.tensors["_root_states"].reshape(3, 2, 2)[1],
            source.tensors["_root_states"].reshape(3, 2, 2)[1],
        ))
        self.assertTrue(torch.equal(
            target.tensors["_dof_state"].reshape(3, 4, 2)[1],
            source.tensors["_dof_state"].reshape(3, 4, 2)[1],
        ))
        self.assertEqual(float(target.tensors["_root_states"][:2].sum()), 0.0)
        self.assertEqual(float(target.tensors["_dof_state"][8:].sum()), 0.0)

    def test_update_where_ignores_lazily_added_source_buffers(self):
        target = TaskBranchSnapshot(
            tensors={"_root_states": torch.zeros(2, 1)},
            factors={"_root_states": 1},
            num_envs=2,
        )
        source = TaskBranchSnapshot(
            tensors={
                "_root_states": torch.ones(2, 1),
                "_lazy_render_cache": torch.full((2, 1), 9.0),
            },
            factors={"_root_states": 1, "_lazy_render_cache": 1},
            num_envs=2,
        )
        target.update_where(source, torch.tensor([True, False]))
        self.assertEqual(target.tensors["_root_states"].flatten().tolist(), [1.0, 0.0])
        self.assertNotIn("_lazy_render_cache", target.tensors)

    def test_update_where_rejects_changed_base_buffer(self):
        target = TaskBranchSnapshot(
            tensors={"_root_states": torch.zeros(2, 1)},
            factors={"_root_states": 1},
            num_envs=2,
        )
        source = TaskBranchSnapshot(tensors={}, factors={}, num_envs=2)
        with self.assertRaisesRegex(ValueError, "missing=.*_root_states"):
            target.update_where(source, torch.ones(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
