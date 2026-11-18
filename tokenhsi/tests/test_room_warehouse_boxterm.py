from copy import deepcopy
import math
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.square_sensor_scene import compute_box_cpa_collision_penalty
from env.tasks.multi_agent.warehouse_sensor import collision, push_box_contact_step
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import PUSH, ROOM, WAREHOUSE, validate_sensor


NAME = 'stage2_sensor_900_room_warehouse_wallterm_boxterm'
BASE = 'stage2_sensor_900_room_warehouse_wallterm'
IDENTITY = torch.tensor([0., 0., 0., 1.])


def test_boxterm_adds_only_the_push_box_keys():
    current, base = native_config(NAME), native_config(BASE)
    train, base_train = native_config(NAME, True), native_config(BASE, True)
    validate_experiment(NAME, current['env'], train)
    scene = current['env']['sensorScene']
    assert (scene['push_box_collision_coefficient'], scene['push_box_contact_termination']) == (.7, .3)
    restored = deepcopy(current)
    restored['experiment']['name'] = BASE
    del restored['env']['sensorScene']['push_box_collision_coefficient']
    del restored['env']['sensorScene']['push_box_contact_termination']
    assert restored == base
    assert train['params']['config']['name'] == 'Stage2Sensor900RoomWarehouseWallTermBoxTerm'
    train['params']['config']['name'] = base_train['params']['config']['name']
    assert train == base_train


@pytest.mark.parametrize('key,value', [('push_box_collision_coefficient', -.1),
    ('push_box_collision_coefficient', float('nan')), ('push_box_contact_termination', 0),
    ('push_box_contact_termination', float('inf'))])
def test_invalid_push_box_values_are_rejected(key, value):
    env = native_config(NAME)['env']
    env['sensorScene'][key] = value
    with pytest.raises(ValueError):
        validate_sensor(env)


def test_push_box_keys_require_the_warehouse():
    env = native_config(NAME)['env']
    del env['sensorScene']['warehouse']
    with pytest.raises(ValueError, match='warehouse scene'):
        validate_sensor(env)


def box_risk(root_vel, box_vel, half=(.3, 1., .75), quat=IDENTITY):
    return compute_box_cpa_collision_penalty(torch.tensor([[[0., 0., .9]]]), torch.tensor([[root_vel]]),
        torch.tensor([[1., 0., .75]]), quat[None], torch.tensor([half]), torch.tensor([box_vel]), .7, .95, 1/30)[0, 0]


def test_box_cpa_uses_the_nearest_face_and_relative_velocity():
    approach = box_risk([1., 0., 0.], [0., 0., 0.])
    assert approach > 0
    torch.testing.assert_close(box_risk([0., 0., 0.], [-1., 0., 0.]), approach)
    assert box_risk([0., 1., 0.], [0., 0., 0.]) == 0
    yaw = torch.tensor([0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
    torch.testing.assert_close(box_risk([1., 0., 0.], [0., 0., 0.], (1., .3, .75), yaw), approach)


def make_task(collision_scale=1.4, limit=9):
    n, m = 2, 2
    box = torch.zeros(n, 3, 13)
    box[:, 2, :3] = torch.tensor([1., 0., .75])
    box[:, 2, 3:7] = IDENTITY
    roots = torch.zeros(n, m, 13)
    roots[..., 2] = .9
    roots[..., 7] = 1.
    return SimpleNamespace(
        num_agents=m, dt=1/30, progress_buf=torch.full((n,), 5), _enable_early_termination=True,
        _scene_ids=torch.tensor([WAREHOUSE, ROOM]), _templates=torch.tensor([[PUSH, 0], [PUSH, 0]]),
        _bridge_slot=torch.tensor([2, 2]), _box_states=box, _box_size=torch.tensor([.6, 2., 1.5]).expand(n, 3, 3),
        _humanoid_root_states=roots, _warehouse={'wall_ttc_discount': .95}, _wall_collision_scale=1.4,
        _warehouse_wall_starts=torch.full((n, 1, 2), 50.), _warehouse_wall_ends=torch.full((n, 1, 2), 51.),
        _push_box_collision_scale=collision_scale, _push_box_contact_limit=limit,
        _push_box_contact_steps=torch.zeros(n, m, dtype=torch.long), _push_box_outcomes=torch.zeros(3),
        _wall_contact_body_ids=[0], _wall_contact_shapes=(torch.zeros(1, 3), torch.tensor([.05]), torch.tensor([0])),
        _rigid_body_pos=torch.tensor([.65, 0., 1.]).expand(n, m, 1, 3), _rigid_body_rot=IDENTITY.expand(n, m, 1, 4),
        _contact_forces=torch.ones(n, m, 1, 3))


def test_push_box_cpa_skips_pushers_and_other_scenes():
    task = make_task()
    agents = collision(task, torch.zeros(2, 2), task._humanoid_root_states[..., :3], .7)
    assert agents[0, 0] == 0 and agents[0, 1] > 0 and not agents[1].any()
    task = make_task(collision_scale=0.)
    assert not collision(task, torch.zeros(2, 2), task._humanoid_root_states[..., :3], .7).any()


def test_non_pusher_contact_terminates_on_the_ninth_step_and_resets_when_released():
    task = make_task()
    ended = [push_box_contact_step(task).tolist() for _ in range(9)]
    assert ended[:8] == [[[False, False], [False, False]]]*8 and ended[8] == [[False, True], [False, False]]
    task._contact_forces = torch.zeros(2, 2, 1, 3)
    assert not push_box_contact_step(task).any() and not task._push_box_contact_steps.any()
    assert task._push_box_outcomes.tolist() == [9., 10., 1.]


def test_without_duration_push_box_contact_is_only_logged():
    task = make_task(limit=None)
    assert not any(push_box_contact_step(task).any() for _ in range(20))
    assert task._push_box_outcomes.tolist() == [20., 20., 0.]
