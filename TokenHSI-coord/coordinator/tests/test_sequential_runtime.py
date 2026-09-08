"""Execute the real adapter and real ms18 steer math against CPU tensor fixtures.

Only the simulator-owning parent is stubbed. This does not assert physics/task
success; it tests the code that selects, installs and exposes policy commands.
"""

import ast
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from coordinator.tests.common import make_state
from coordinator.tests.test_sequential_bridge import straight

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "TokenHSI-masteer/tokenhsi/env/tasks/adapt_interaction_skills"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = load_source("coord_test_steer_path", ROOT / "TokenHSI-masteer/tokenhsi/utils/steer_path.py")

# Exercise the exact existing ms18 observation math, without importing Isaac Gym.
tree = ast.parse((TASKS / "humanoid_ma_steer_carry.py").read_text())
native = next(node for node in tree.body if isinstance(node, ast.ClassDef))
methods = [node for node in native.body if isinstance(node, ast.FunctionDef)
           and node.name in ("_steer_obs", "_m_at", "_handheld_score")]
namespace = {"torch": torch, "sp": sp}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(TASKS), "exec"), namespace)


class TensorSequentialBase:
    _steer_obs = namespace["_steer_obs"]
    _m_at = namespace["_m_at"]
    _handheld_score = namespace["_handheld_score"]

    def __init__(self, *args):
        self.device, self.num_envs, self.num_agents, self._rows = "cpu", 2, 2, 4
        self.steer_k, self.steer_m_nom, self.steer_clip, self.steer_zero = 6, 2.4, True, False
        self.dt = 1 / 30
        state = make_state(2)
        self._humanoid_root_states = torch.zeros(2, 2, 13)
        self._humanoid_root_states[..., :2] = state.root_xy
        self._humanoid_root_states[..., 6] = 1
        self._humanoid_root_states[..., 7:9] = state.root_vel_xy
        self._box_states = torch.zeros(2, 2, 13)
        self._box_states[..., :3] = state.box_xyz
        self._box_states[..., 6] = 1
        self._rigid_body_pos = torch.zeros(2, 2, 2, 3)
        self._key_body_ids = torch.tensor([0, 1])
        self._box_lib = types.SimpleNamespace(_box_size=torch.full((4, 3), 0.5))
        self._initial_box_bottom_z = torch.full((4,), 0.05)
        self._box_tar_pos = torch.cat((state.goal_xy, torch.full((2, 2, 1), 0.25)), -1).reshape(4, 3)
        self._gt_path = torch.full((4, 320, 2), -7.)
        self._mscale = torch.full((4, 320), 0.5)
        self._s_end = torch.full((4,), 12.)
        self._arc_root, self._arc_box, self._prev_arc = (torch.zeros(4) for _ in range(3))
        self._stack_phase = torch.tensor([0, 3])
        self._carry_rehearsal = torch.zeros(2, dtype=torch.bool)
        self.progress_buf = torch.zeros(2, dtype=torch.long)
        self.native_fallbacks = []

    def steer_dim(self):
        return 12

    def agent_rows(self, ids):
        return (ids[:, None] * 2 + torch.arange(2)[None]).reshape(-1)

    def humanoid_rows(self, tensor):
        return tensor.flatten(0, 1)

    def progress_rows(self):
        return self.progress_buf.repeat_interleave(2)

    def _compute_task_obs(self, env_ids=None):
        ids = torch.arange(2) if env_ids is None else env_ids
        rows = self.agent_rows(ids)
        return torch.cat((torch.full((len(rows), 21), 11.), self._steer_obs(rows),
                          torch.full((len(rows), 84), 22.)), -1)

    def _compute_observations(self, env_ids=None):
        self.last_obs = self._compute_task_obs(env_ids)

    def _reset_envs(self, env_ids):
        rows = self.agent_rows(env_ids)
        self._gt_path[rows] = -7
        self.progress_buf[env_ids] = 0
        self._compute_observations(env_ids)

    def _reset_steer_to(self, rows, goal):
        self.native_fallbacks.extend(rows.tolist())
        self._gt_path[rows] = -9


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(learned_priority=False)
        self.calls, self.invalid, self.speed = 0, False, 0.6

    def forward(self, state):
        self.calls += 1
        out = straight(state)
        out["speed"][:] = self.speed
        if self.invalid:
            out["path_world"][:] = float("nan")
        return out


class SequentialRuntimeTest(unittest.TestCase):
    def setUp(self):
        parent_name = "env.tasks.adapt_interaction_skills.humanoid_ma_sequential_stack_carry"
        parent = types.ModuleType(parent_name)
        parent.HumanoidMASequentialStackCarry = TensorSequentialBase
        utils = types.ModuleType("tokenhsi.utils")
        utils.steer_path = sp
        with patch.dict(sys.modules, {parent_name: parent, "tokenhsi.utils": utils}):
            self.module = load_source("coord_stack_test_runtime", TASKS / "humanoid_ma_coord_sequential_stack.py")
        self.planner = Planner()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        dummy = Path(self.tmp.name) / "coord.pth"
        dummy.touch()
        self.envpatch = patch.dict(os.environ, {"COORD_CKPT": str(dummy), "COORD_REPLAN_STEPS": "6",
                                                "COORD_COMMAND_ACCEL": "0.75"})
        self.envpatch.start()
        self.addCleanup(self.envpatch.stop)
        with patch.object(self.module, "load_planner", return_value=(self.planner, {"schema_version": "test"})):
            self.task = self.module.HumanoidMACoordSequentialStack({"env": {"numAgents": 2}},
                                                                 None, None, "cpu", 0, True)

    def test_only_active_rows_change_and_other_tokens_are_preserved(self):
        task = self.task
        self.assertEqual(task._coord_applied.tolist(), [True, False, False, True])
        self.assertTrue((task._gt_path[[1, 2]] == -7).all())
        self.assertEqual(task.last_obs.shape, (4, 117))  # + self223 = policy340
        self.assertTrue((task.last_obs[:, :21] == 11).all())
        self.assertTrue((task.last_obs[:, 33:] == 22).all())
        self.assertTrue(torch.isfinite(task.last_obs).all())
        self.assertEqual(task._box_tar_pos[:, 2].tolist(), [0.25] * 4)

    def test_subenv_replan_does_not_touch_other_environment(self):
        task = self.task
        old = task._gt_path[:2].clone()
        task.progress_buf[1] = 6
        task._compute_task_obs(torch.tensor([1]))
        torch.testing.assert_close(task._gt_path[:2], old)
        self.assertEqual(task._coord_tick.tolist(), [0, 6])

    def test_reset_is_independent_and_refreshes_first_observation(self):
        task = self.task
        task.progress_buf[:] = 12
        before = task._gt_path[:2].clone()
        task._reset_envs(torch.tensor([1]))
        self.assertEqual(task.progress_buf.tolist(), [12, 0])
        torch.testing.assert_close(task._gt_path[:2], before)
        self.assertEqual(task.last_obs.shape, (2, 117))
        self.assertTrue(task._coord_applied[3])
        self.assertTrue((task._gt_path[2] == -7).all())

    def test_repeated_observation_same_tick_does_not_replan_or_accelerate(self):
        task = self.task
        calls, speed = self.planner.calls, task._coord_speed.clone()
        task._compute_task_obs()
        self.assertEqual(self.planner.calls, calls)
        torch.testing.assert_close(task._coord_speed, speed)
        task.progress_buf[:] += 1
        task._compute_task_obs()
        self.assertAlmostEqual(float(task._coord_speed[0] - speed[0]), -0.75 / 30, places=6)

    def test_speed_changes_real_ms18_steer_window(self):
        task = self.task
        task._coord_speed[0] = 0.375
        slow = task._steer_obs(torch.tensor([0]))
        task._coord_speed[0] = 1.5
        fast = task._steer_obs(torch.tensor([0]))
        self.assertGreater(float(fast.norm()), float(slow.norm()) * 2)

    def test_invalid_replan_keeps_previous_but_changed_goal_uses_fallback(self):
        task = self.task
        before = task._gt_path[0].clone()
        self.planner.invalid = True
        task.progress_buf[:] += 6
        task._compute_task_obs()
        torch.testing.assert_close(task._gt_path[0], before)
        self.assertEqual(task.native_fallbacks, [])
        task._box_tar_pos[0, 0] += 1
        task._compute_task_obs(torch.tensor([0]))
        self.assertEqual(task.native_fallbacks, [0])
        self.assertFalse(task._coord_applied[0])

    def test_retreat_removes_planner_ownership_and_a2_replans_immediately(self):
        task = self.task
        task._stack_phase[0] = 2
        task._gt_path[0] = 123  # Native phase transition installed retreat path.
        calls = self.planner.calls
        task._compute_task_obs(torch.tensor([0]))
        self.assertEqual(self.planner.calls, calls)
        self.assertTrue((task._gt_path[0] == 123).all())
        self.assertFalse(task._coord_applied[:2].any())
        task._stack_phase[0] = 3
        task._compute_task_obs(torch.tensor([0]))
        self.assertTrue(task._coord_applied[1])
        self.assertTrue((task._gt_path[0] == 123).all())

    def test_c13_priority_contract_and_sequential_phase_gate_coexist(self):
        task = self.task
        task._coord_random_priority = True
        task._coord_priority[:] = 0
        task.progress_buf[:] += 6
        task._compute_task_obs()
        # C13 priority A1 is nominal; A2's learned speed is still used when active.
        torch.testing.assert_close(task._mscale[0], torch.ones(320))
        torch.testing.assert_close(task._mscale[3], torch.full((320,), 0.4))
        self.assertEqual(task._coord_applied.tolist(), [True, False, False, True])
        task._coord_priority_locked[:] = True
        task._coord_reset(torch.tensor([1]))
        self.assertTrue(task._coord_priority_locked[0])
        self.assertFalse(task._coord_priority_locked[1])

    def test_measured_held_requires_hands_and_lift(self):
        task = self.task
        task._humanoid_root_states[..., :2] = task._box_states[..., :2]
        task._box_states[..., 2] = 0.8
        self.assertFalse(task.coord_state(torch.tensor([1])).held.any())
        task._rigid_body_pos[:] = task._box_states[..., None, :3]
        self.assertTrue(task.coord_state(torch.tensor([1])).held.all())
