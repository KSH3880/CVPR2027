"""CPU tensors + fake Gym API. Never imports real isaacgym or initializes CUDA."""
import sys
import types
import unittest
from unittest.mock import patch

import torch
from examples.make_synthetic_fixture import fixture
from sct_scheduler.isaac_adapter import InitialReleaseFreeze


class FakeGym:
    def __init__(self):
        self.submissions = []

    def set_actor_root_state_tensor_indexed(self, sim, state, ids, count):
        self.submissions.append(('root', ids.tolist()))

    def set_dof_state_tensor_indexed(self, sim, state, ids, count):
        self.submissions.append(('dof', ids.tolist()))


def fake_task():
    task = types.SimpleNamespace(num_envs=2, num_agents=2, device='cuda:0', dt=.05, sim=None, gym=FakeGym())
    roots = torch.zeros(2, 3, 13)
    roots[..., 6] = 1
    task._root_states = roots.view(-1, 13)
    task._humanoid_root_states = roots[:, :2]  # Non-contiguous actor view, not reshape copy.
    task._humanoid_actor_ids = torch.tensor([[0, 1], [3, 4]], dtype=torch.int32)
    task._dof_state = torch.zeros(2, 2, 3, 2)
    task._dof_pos = task._dof_state[..., 0]
    task._dof_vel = task._dof_state[..., 1]
    bodies = torch.zeros(2, 2, 2, 13)
    bodies[..., 6] = 1
    task._kinematic_humanoid_rigid_body_states = bodies
    task._rigid_body_pos, task._rigid_body_rot = bodies[..., :3], bodies[..., 3:7]
    task._rigid_body_vel, task._rigid_body_ang_vel = bodies[..., 7:10], bodies[..., 10:13]
    task._refresh_sim_tensors = lambda: None
    task._physics_step = lambda: task._refresh_sim_tensors()
    task._action_to_pd_targets = lambda actions: actions.reshape(2, 2, 3)
    return task


class AdapterTests(unittest.TestCase):
    def test_projection_release_batch_and_hook_cleanup(self):
        request, response = fixture()
        task = fake_task()
        original = task._refresh_sim_tensors
        tensor, arange = torch.tensor, torch.arange
        def cpu_tensor(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return tensor(*args, **kwargs)
        def cpu_arange(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return arange(*args, **kwargs)
        fake_isaac = types.ModuleType('isaacgym')
        fake_isaac.gymtorch = types.SimpleNamespace(unwrap_tensor=lambda x: x)
        with patch.dict(sys.modules, {'isaacgym': fake_isaac}), \
                patch('sct_scheduler.isaac_adapter.check_gpu_mapping'), \
                patch('torch.tensor', side_effect=cpu_tensor), patch('torch.arange', side_effect=cpu_arange):
            controller = InitialReleaseFreeze(task, [request, request], [response, response],
                                               [['A', 'B'], ['B', 'A']], physical_gpu=6)
            self.assertEqual(controller.frozen.tolist(), [[True, False], [False, True]])
            self.assertIn(('root', [0, 1, 2, 3, 4, 5]), task.gym.submissions)
            task._humanoid_root_states[..., 0] = 9
            task._dof_pos[:] = 7
            task._dof_vel[:] = 5
            task._physics_step()
            self.assertEqual(task._humanoid_root_states[..., 0].tolist(), [[0, 9], [9, 0]])
            self.assertEqual(task._root_states[:, 0].tolist(), [0, 9, 0, 9, 0, 0])
            pd = task._action_to_pd_targets(torch.ones(4, 3))
            self.assertEqual(pd[:, :, 0].tolist(), [[0, 1], [1, 0]])
            controller.before_action(39)
            self.assertTrue(controller.frozen.any())
            controller.before_action(40)
            self.assertFalse(controller.frozen.any())
            task._humanoid_root_states[..., 0] = 12
            task._physics_step()
            self.assertTrue((task._humanoid_root_states[..., 0] == 12).all())
            controller.before_action(41)
            self.assertFalse(controller.frozen.any())
            proof = controller.close()
            controller.close()
            self.assertIs(task._refresh_sim_tensors, original)
            self.assertEqual(len(proof['release_events'][0]), 2)
            self.assertFalse(hasattr(task, '_sct_initial_release_adapter'))


if __name__ == '__main__':
    unittest.main()
