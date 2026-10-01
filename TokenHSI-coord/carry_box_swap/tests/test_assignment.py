import unittest
from types import SimpleNamespace
import torch
from carry_box_swap.assignment import assigned_inputs, assignment_rows


class BoxAssignmentTest(unittest.TestCase):
    def task(self):
        storage = torch.arange(2*5*13).reshape(2, 5, 13).float()
        boxes = storage[:, 2:4]
        return SimpleNamespace(num_envs=2, box_assignment=torch.tensor([[0, 1], [1, 0]]),
            _box_states=boxes, _box_tar_pos=torch.arange(12).reshape(4, 3).float(),
            _prev_box_pos=boxes[..., :3].reshape(4, 3).clone(),
            _box_lib=SimpleNamespace(_box_size=torch.arange(12).reshape(4, 3).float(),
                                    _box_bps=torch.arange(96).reshape(4, 8, 3).float()),
            humanoid_rows=lambda t: t.reshape(4, -1))

    def test_permutation_preserves_every_box_exactly_once(self):
        self.assertEqual(assignment_rows(torch.tensor([[0, 1], [1, 0]])).tolist(), [0, 1, 3, 2])
        for bad in (torch.tensor([[0, 0]]), torch.tensor([[2, 1]])):
            with self.assertRaises(ValueError): assignment_rows(bad)

    def test_positions_velocity_shapes_and_goal_follow_assigned_box(self):
        task = self.task(); boxes = task._box_states; copied = boxes.clone()
        goals = task._box_tar_pos; previous = task._prev_box_pos; size = task._box_lib._box_size
        bps = task._box_lib._box_bps; index = torch.tensor([0, 1, 3, 2])
        with assigned_inputs(task):
            self.assertTrue(torch.equal(task._box_states.reshape(4, 13), copied.reshape(4, 13)[index]))
            self.assertTrue(torch.equal(task._box_tar_pos, goals[index]))
            self.assertTrue(torch.equal(task._prev_box_pos, previous[index]))
            self.assertTrue(torch.equal(task._box_lib._box_size, size[index]))
            self.assertTrue(torch.equal(task._box_lib._box_bps, bps[index]))
        self.assertIs(task._box_states, boxes)
        self.assertTrue(torch.equal(boxes, copied))
        self.assertIs(task._box_tar_pos, goals)
        self.assertIs(task._box_lib._box_size, size)

    def test_agent_goals_can_remain_fixed_and_exception_restores_aliases(self):
        task = self.task(); original = task._box_states; goal = task._box_tar_pos
        with self.assertRaises(RuntimeError):
            with assigned_inputs(task, goals_follow_box=False):
                self.assertIs(task._box_tar_pos, goal)
                raise RuntimeError("probe failed")
        self.assertIs(task._box_states, original)
        self.assertIs(task._box_tar_pos, goal)


if __name__ == "__main__": unittest.main()
