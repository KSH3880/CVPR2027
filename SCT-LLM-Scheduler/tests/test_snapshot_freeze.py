"""Verify suspension/resumption semantics without CUDA or a simulator."""
import sys
import types
import unittest
from unittest.mock import patch

import torch
from examples.make_synthetic_fixture import fixture
from sct_scheduler.snapshot_freeze import SnapshotFreeze
from test_adapter import fake_task


class SnapshotFreezeTests(unittest.TestCase):
    def test_resume_initial_motion_once_in_batched_actor_order(self):
        request, response = fixture()
        task = fake_task()
        task.device = 'cpu'
        task._humanoid_root_states[..., 7] = .7
        task._dof_vel[:] = .3
        task._kinematic_humanoid_rigid_body_states[..., 7] = .7
        refresh = task._refresh_sim_tensors
        gym = types.ModuleType('isaacgym')
        gym.gymtorch = types.SimpleNamespace(unwrap_tensor=lambda x: x)
        with patch.dict(sys.modules, {'isaacgym': gym}), \
                patch('sct_scheduler.snapshot_freeze.check_gpu_mapping'):
            freeze = SnapshotFreeze(task, [request]*2, [response]*2,
                                    [['A', 'B'], ['B', 'A']], physical_gpu=6)
            held = freeze.frozen.clone()
            self.assertEqual(held.tolist(), [[True, False], [False, True]])
            self.assertTrue((task._dof_vel[held] == 0).all())
            self.assertTrue((task._dof_vel[~held] == .3).all())
            task._humanoid_root_states[held] += 1
            task._physics_step()
            self.assertTrue((task._humanoid_root_states[held][..., 7:] == 0).all())
            freeze.before_action(39)
            self.assertTrue(freeze.frozen.any())
            freeze.before_action(40)
            self.assertFalse(freeze.frozen.any())
            self.assertTrue((task._dof_vel[held] == .3).all())
            self.assertTrue((task._humanoid_root_states[held][..., 7] == .7).all())
            self.assertTrue((task._rigid_body_vel[held][..., 0] == .7).all())
            self.assertEqual(task.gym.submissions[-2:], [('root', [0, 4]), ('dof', [0, 4])])
            task._dof_vel[:] = 2
            freeze.before_action(41)
            task._physics_step()
            self.assertTrue((task._dof_vel == 2).all())
            proof = freeze.close()
            freeze.close()
            self.assertEqual(proof['max_restoration_error'], 0)
            self.assertIs(task._refresh_sim_tensors, refresh)
            self.assertFalse(hasattr(task, '_sct_initial_release_adapter'))
