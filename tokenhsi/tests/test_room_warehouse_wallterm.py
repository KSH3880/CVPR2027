from copy import deepcopy
import math

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.sensor_stage2_observation import SensorStage2Observation, WALL_CONTACT_BODIES
from env.tasks.multi_agent.square_sensor_scene import body_box_contact
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import CLIFF, ROOM, WAREHOUSE, validate_sensor


NAME = 'stage2_sensor_900_room_warehouse_wallterm'
BASE = 'stage2_sensor_900_room_warehouse'


def test_wallterm_adds_only_the_termination_duration():
    current, base = native_config(NAME), native_config(BASE)
    train, base_train = native_config(NAME, True), native_config(BASE, True)
    validate_experiment(NAME, current['env'], train)
    assert current['env']['sensorScene']['wall_contact_termination'] == .3
    restored = deepcopy(current)
    restored['experiment']['name'] = BASE
    del restored['env']['sensorScene']['wall_contact_termination']
    assert restored == base
    assert train['params']['config']['name'] == 'Stage2Sensor900RoomWarehouseWallTerm'
    train['params']['config']['name'] = base_train['params']['config']['name']
    assert train == base_train


@pytest.mark.parametrize('duration', [0, -.1, float('nan'), float('inf'), True])
def test_invalid_duration_is_rejected(duration):
    env = native_config(NAME)['env']
    env['sensorScene']['wall_contact_termination'] = duration
    with pytest.raises(ValueError):
        validate_sensor(env)


def test_hands_and_feet_are_not_wall_contact_bodies():
    assert not {'right_hand', 'left_hand', 'right_foot', 'left_foot'} & set(WALL_CONTACT_BODIES)
    assert len(WALL_CONTACT_BODIES) == 11


SPHERE = (torch.zeros(1, 3), torch.tensor([.05]), torch.tensor([0]))
IDENTITY = torch.tensor([0., 0., 0., 1.])


def test_contact_needs_force_and_a_shape_touching_the_wall_frame():
    yaw = math.pi/2
    centers = torch.tensor([[[1., 0., 1.25]]]*3+[[[0., 1., 1.25]]])
    quats = torch.tensor([[[0., 0., 0., 1.]]]*3+[[[0., 0., math.sin(yaw/2), math.cos(yaw/2)]]])
    halves = torch.tensor([[[.2, 5., 1.25]]]).expand(4, -1, -1)
    body = torch.tensor([[.73, 0., 1.], [.73, 0., 1.], [.6, 0., 1.], [0., .73, 1.]])[:, None, None]
    force = torch.tensor([[1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])[:, None, None]
    touching = body_box_contact(body, IDENTITY.expand(4, 1, 1, 4), force, SPHERE, centers, quats, halves, .03)
    assert touching[:, 0].tolist() == [True, False, False, True]


def test_offset_shapes_follow_the_body_rotation():
    shapes = (torch.tensor([[0., 0., .175]]), torch.tensor([.05]), torch.tensor([0]))
    pitch = torch.tensor([0., math.sin(math.pi/4), 0., math.cos(math.pi/4)])
    rot = torch.stack((IDENTITY, pitch))[:, None, None]
    body = torch.tensor([[.6, 0., 1.]]).expand(2, 1, 1, 3)
    wall = (torch.tensor([[[1., 0., 1.25]]]).expand(2, -1, -1), IDENTITY.expand(2, 1, 4),
            torch.tensor([[[.2, 5., 1.25]]]).expand(2, -1, -1))
    touching = body_box_contact(body, rot, torch.ones(2, 1, 1, 3), shapes, *wall, .03)
    assert touching[:, 0].tolist() == [False, True]


def make_task(limit):
    class Task(SensorStage2Observation):
        pass

    task = Task()
    n, m = 3, 1
    task.num_agents = m
    task._scene_names = ('cliff', 'bridge', 'room', 'warehouse')
    task._scene_ids = torch.tensor([ROOM, CLIFF, WAREHOUSE])
    task.progress_buf = torch.full((n,), 5)
    task._enable_early_termination = True
    task._wall_contact_limit = limit
    task._wall_contact_steps = torch.zeros(n, m, dtype=torch.long)
    task._wall_outcomes = torch.zeros(4, 3)
    task._wall_contact_body_ids = [0]
    task._wall_contact_shapes = SPHERE
    task._static_centers = torch.zeros(n, 4, 6, 3)
    task._static_centers[:, :, 1:] = torch.tensor([10., 10., 1.25])
    task._static_centers[:, :, 1] = torch.tensor([1., 0., 1.25])
    task._static_quat = torch.tensor([0., 0., 0., 1.]).expand(n, 4, 6, 4)
    task._static_half = torch.tensor([.2, 5., 1.25]).expand(n, 4, 6, 3)
    # body 0 is a counted body, body 1 stands in for an excluded hand
    task._rigid_body_pos = torch.tensor([[.73, 0., 1.]]).expand(n, m, 2, 3).clone()
    task._rigid_body_rot = IDENTITY.expand(n, m, 2, 4)
    task._contact_forces = torch.zeros(n, m, 2, 3)
    task._contact_forces[0, 0, 0, 0] = task._contact_forces[1, 0, 0, 0] = 1.
    task._contact_forces[2, 0, 1, 0] = 1.
    task._rigid_body_pos[2, 0, 0, 0] = -3.
    return task


def test_wall_contact_terminates_on_the_ninth_step_and_resets_when_released():
    task = make_task(math.ceil(.3/(1/30)-1e-6))
    assert task._wall_contact_limit == 9
    walled = [task._wall_contact_step()[:, 0].tolist() for _ in range(9)]
    assert walled[:8] == [[False]*3]*8 and walled[8] == [True, False, False]
    assert task._wall_contact_steps[:, 0].tolist() == [9, 0, 0]
    task._contact_forces.zero_()
    assert not task._wall_contact_step().any() and not task._wall_contact_steps.any()
    assert task._wall_outcomes[ROOM].tolist() == [9., 10., 1.]
    assert task._wall_outcomes[CLIFF].tolist() == [0., 10., 0.]
    assert task._wall_outcomes[WAREHOUSE].tolist() == [0., 10., 0.]


def test_without_duration_contact_is_only_logged():
    task = make_task(None)
    assert not any(task._wall_contact_step().any() for _ in range(20))
    assert task._wall_outcomes[ROOM].tolist() == [20., 20., 0.]
