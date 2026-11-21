import isaacgym
import pytest
import torch

from env.tasks.multi_agent.square_sensor_scene import yaw_quat
from env.tasks.multi_agent.warehouse_close import combine, risks
from test_warehouse_close import make_task
from utils.experiments import native_config
from utils.sensor_stage2_spec import validate_sensor


def push_task():
    task = make_task()
    task._warehouse['wall_proximity']['push_box'] = True
    task._bridge_slot = torch.tensor([0])
    task._templates = torch.tensor([[5, 5, 1, 1]])
    task._box_states[0, 0, :3] = torch.tensor([0., 0., 1.])
    task._box_size[0, 0] = torch.tensor([2., 1., 2.])
    task._rigid_body_pos[:] = torch.tensor([1.25, 0., 1.])
    task.relation_runtime.own_success.zero_()
    return task


def test_body_surface_clearance_and_pusher_exemption():
    task = push_task()
    human, payload = risks(task, torch.tensor([0]))
    torch.testing.assert_close(human, torch.tensor([[0., 0., .5, .5]]), atol=3e-6, rtol=0)
    assert not payload.any()


@pytest.mark.parametrize('change,expected', [('translate', 1.), ('rotate', 0.), ('height', 0.)])
def test_current_obstacle_pose_is_used(change, expected):
    task = push_task()
    if change == 'translate':
        task._box_states[0, 0, 0] = .1
    elif change == 'rotate':
        task._box_states[0, 0, 3:7] = yaw_quat(torch.tensor([torch.pi / 2]))[0]
    else:
        task._box_states[0, 0, 2] = 4.
    human, _ = risks(task, torch.tensor([0]))
    torch.testing.assert_close(human, torch.tensor([[0., 0., expected, expected]]))


def test_only_current_nonpush_payload_is_penalized():
    task = push_task()
    task._rigid_body_pos[:] = torch.tensor([2., 0., 1.])
    task._box_states[0, 1:, :3] = torch.tensor([1.3, 0., 1.])
    task.relation_runtime.own_success[:] = True
    human, payload = risks(task, torch.tensor([0]))
    assert not human.any()
    torch.testing.assert_close(payload, torch.tensor([[0., 0., .5, 0.]]), atol=3e-6, rtol=0)
    task.relation_runtime.own_success.zero_()
    assert not risks(task, torch.tensor([0]))[1].any()


def test_no_push_scene_and_disabled_option_add_no_box_penalty():
    task = push_task()
    task._templates[:] = 1
    task.relation_runtime.own_success[:] = True
    assert not risks(task, torch.tensor([0]))[0].any()
    task.relation_runtime.own_success.zero_()
    task._templates[0, :2] = 5
    task._warehouse['wall_proximity'].pop('push_box')
    assert not risks(task, torch.tensor([0]))[0].any()


def test_between_wall_and_box_uses_max_and_preserves_wall_cost_for_pushers():
    task = push_task()
    task._rigid_body_pos[:] = torch.tensor([3.75, 3., 1.])
    task._box_states[0, 0, :3] = torch.tensor([2.5, 3., 1.])
    penalty = combine(task, torch.tensor([0]), torch.zeros(1, 4)) * .5
    torch.testing.assert_close(penalty, torch.full((1, 4), .35), atol=3e-6, rtol=0)
    task._box_states[0, 0, 0] += .03
    penalty = combine(task, torch.tensor([0]), torch.tensor([[0., 0., .9, 0.]])) * .5
    torch.testing.assert_close(penalty, torch.tensor([[.35, .35, .56, .56]]), atol=3e-6, rtol=0)


def test_partial_batch_uses_each_scenes_physical_box_and_roles():
    task = push_task()
    task._warehouse['wall_proximity']['payload'] = False
    for key, value in list(vars(task).items()):
        if isinstance(value, torch.Tensor) and value.shape[0] == 1:
            setattr(task, key, value.repeat(3, *([1] * (value.ndim - 1))))
    task.num_envs = 3
    task._bridge_slot = torch.tensor([0, 1, 2])
    task._box_states[..., :3] = 20.
    for env, slot in enumerate(task._bridge_slot):
        task._box_states[env, slot, :3] = torch.tensor([0., 0., 1.])
        task._box_size[env, slot] = torch.tensor([2., 1., 2.])
    task._templates[2] = torch.tensor([1, 5, 1, 5])
    human, _ = risks(task, torch.tensor([2, 0]))
    torch.testing.assert_close(human, torch.tensor([[.5, 0., .5, 0.], [0., 0., .5, .5]]), atol=3e-6, rtol=0)


def test_push_box_option_requires_boolean():
    env = native_config('warehouse_close_wall_box_proximity')['env']
    env['sensorScene']['warehouse']['wall_proximity']['push_box'] = 1
    with pytest.raises(ValueError, match='push_box'):
        validate_sensor(env)
