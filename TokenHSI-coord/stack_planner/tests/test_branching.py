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


if __name__ == "__main__":
    unittest.main()
