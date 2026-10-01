from __future__ import annotations

import unittest
import ast
import os
from pathlib import Path
from unittest.mock import patch

import torch

from carry_planner.layout import converging_goal_xy, two_agent_layout_slots


def _layout_harness_types():
    """Run the real reset layout methods without importing Isaac Gym."""
    coord = Path(__file__).resolve().parents[2]
    definitions = []
    for file, name, methods, bases in (
        (coord / "tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py",
         "HumanoidTrajSitCarryClimb", {"apply_layout", "_layout_slot_indices"}, []),
        (coord / "carry_planner/env_adapter.py", "HumanoidMACarryPlannerTrain",
         {"_layout_slot_indices"}, [ast.Name(id="HumanoidTrajSitCarryClimb", ctx=ast.Load())]),
    ):
        tree = ast.parse(file.read_text())
        original = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == name)
        definitions.append(ast.ClassDef(
            name=name, bases=bases, keywords=[], decorator_list=[],
            body=[node for node in original.body
                  if isinstance(node, ast.FunctionDef) and node.name in methods],
        ))
    namespace = {"torch": torch, "os": os,
                 "two_agent_layout_slots": two_agent_layout_slots}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 "<real-layout-methods>", "exec"), namespace)
    return namespace["HumanoidTrajSitCarryClimb"], namespace["HumanoidMACarryPlannerTrain"]


class RandomAgentSlotTest(unittest.TestCase):
    def make_task(self, task_type, agents=2):
        task = task_type()
        task.num_agents, task.device = agents, torch.device("cpu")
        task._carry_randomize_agent_slots = True
        task._carry_reset_random_height = True
        # Strided views represent actor tensors with other actors between envs.
        task._humanoid_root_states = torch.arange(3 * 4 * 13).float().reshape(3, 4, 13)[:, :agents]
        task._initial_humanoid_root_states = torch.zeros(3, agents, 13)
        if agents == 2:
            task._initial_humanoid_root_states[:, 0, 0] = -1.5
            task._initial_humanoid_root_states[:, 1, 0] = 1.5
        task._box_states = task._humanoid_root_states.clone()
        task._box_states[..., :2] += torch.tensor([0.3, -0.2])
        task._platform_states = task._box_states.clone()
        task._tar_platform_states = task._box_states.clone()
        task._box_tar_pos = torch.arange(3 * agents * 3).float().reshape(3 * agents, 3)
        task.agent_axis = lambda tensor: tensor
        task.agent_rows = lambda ids: (ids[:, None] * agents + torch.arange(agents)).reshape(-1)
        return task

    def test_random_slots_cover_both_orders_and_disabled_keeps_original(self):
        base, carry = _layout_harness_types()
        task = self.make_task(carry)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            slots = task._layout_slot_indices(torch.arange(4096))
        self.assertTrue(torch.equal(slots.sort(dim=1).values, torch.tensor([0, 1]).expand(4096, -1)))
        fraction = float((slots[:, 0] == 1).float().mean())
        self.assertTrue(0.45 < fraction < 0.55)
        task._carry_randomize_agent_slots = False
        self.assertTrue(torch.equal(task._layout_slot_indices(torch.arange(3)), torch.tensor([[0, 1]] * 3)))
        single = self.make_task(base, agents=1)
        self.assertTrue(torch.equal(single._layout_slot_indices(torch.arange(3)), torch.zeros(3, 1, dtype=torch.long)))

    def test_actual_partial_reset_swaps_slots_and_preserves_grasp_and_support(self):
        _, carry = _layout_harness_types()
        task = self.make_task(carry)
        task._layout_slot_indices = lambda ids: two_agent_layout_slots(torch.tensor([True, False]))
        fields = ("_humanoid_root_states", "_box_states", "_platform_states", "_tar_platform_states")
        before = {name: getattr(task, name).clone() for name in fields}
        target_before = task._box_tar_pos.clone()
        ids = torch.tensor([2, 0])
        with patch.dict(os.environ, {"MA_LAYOUT": "side", "MA_LAYOUT_L": "12"}):
            task.apply_layout(ids)
        expected = torch.tensor([[[0., -6.], [-6., 0.]], [[-6., 0.], [0., -6.]]])
        self.assertTrue(torch.equal(task._humanoid_root_states[ids, :, :2], expected))
        goals = task._box_tar_pos.reshape(3, 2, 3)
        self.assertTrue(torch.equal(goals[ids, :, :2], -expected))
        self.assertTrue(torch.equal(goals[..., 2], target_before.reshape(3, 2, 3)[..., 2]))
        for name in fields:
            self.assertTrue(torch.equal(getattr(task, name)[..., 2:], before[name][..., 2:]))
            self.assertTrue(torch.equal(getattr(task, name)[1], before[name][1]))
        self.assertTrue(torch.equal(task._box_tar_pos.reshape(3, 2, 3)[1], target_before.reshape(3, 2, 3)[1]))
        self.assertTrue(torch.allclose(
            task._box_states[..., :2] - task._humanoid_root_states[..., :2],
            before["_box_states"][..., :2] - before["_humanoid_root_states"][..., :2],
        ))
        self.assertTrue(torch.equal(task._platform_states[ids, :, :2], task._box_states[ids, :, :2]))
        self.assertTrue(torch.equal(task._tar_platform_states[ids, :, :2], goals[ids, :, :2]))
        # The subsequent convergence generator remains valid in either order.
        _, feasible = converging_goal_xy(task._box_states[ids, :, :2], torch.zeros(2, 2), torch.ones(2, 2, 2), 0.25)
        self.assertTrue(bool(feasible.all()))

    def test_free_layout_keeps_random_reset_untouched(self):
        _, carry = _layout_harness_types()
        task = self.make_task(carry)
        before = task._humanoid_root_states.clone()
        with patch.dict(os.environ, {"MA_LAYOUT": ""}):
            task.apply_layout(torch.arange(3))
        self.assertTrue(torch.equal(task._humanoid_root_states, before))


class ConvergingGoalLayoutTest(unittest.TestCase):
    def test_goals_cross_then_preserve_box_clearance(self):
        crossing = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
        box = torch.tensor([
            [[-2.0, -2.0], [1.0, -5.0]],
            [[-6.0, 4.0], [-3.0, 1.0]],
        ])
        size = torch.tensor([
            [[0.8, 0.6], [0.6, 0.6]],
            [[1.0, 0.5], [0.5, 0.5]],
        ])
        margin = 0.25
        goal, feasible = converging_goal_xy(
            box, crossing, size, margin,
        )
        expected = 0.5 * size.norm(dim=-1).sum(dim=-1) + margin
        self.assertTrue(feasible.all())
        self.assertTrue(torch.allclose(
            (goal[:, 0] - goal[:, 1]).norm(dim=-1), expected,
        ))
        # Goals lie beyond the shared crossing along both incoming box rays.
        incoming = crossing[:, None] - box
        outgoing = goal - crossing[:, None]
        cross2d = incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]
        self.assertTrue(torch.allclose(cross2d, torch.zeros_like(cross2d)))
        self.assertTrue(((incoming * outgoing).sum(dim=-1) > 0.0).all())

    def test_nearly_parallel_box_rays_are_rejected(self):
        goal, feasible = converging_goal_xy(
            torch.tensor([[[-2.0, 0.0], [-3.0, 0.01]]]),
            torch.zeros(1, 2), torch.ones(1, 2, 2), 0.25,
        )
        self.assertEqual(goal.shape, (1, 2, 2))
        self.assertFalse(bool(feasible[0]))

    def test_invalid_margin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            converging_goal_xy(
                torch.zeros(1, 2, 2), torch.zeros(1, 2),
                torch.ones(1, 2, 2), -0.1,
            )


if __name__ == "__main__":
    unittest.main()
