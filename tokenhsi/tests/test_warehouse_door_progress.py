from copy import deepcopy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.warehouse_door_progress import reward
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.sensor_stage2_spec import WAREHOUSE, ROOM


BASE = 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs'
NAME = BASE+'_doorprogress'


def make_task(n=1, yaw=0.):
    m = 4
    half = torch.full((n, 4, 2, 2), 4.)
    static = torch.zeros(n, 4, 6, 3)
    static[:, WAREHOUSE, 4, 1] = 1.3  # doorway half width = 4 + .4 - 2*1.3 = 1.8
    return SimpleNamespace(num_envs=n, num_agents=m, device='cpu',
        _humanoid_root_states=torch.zeros(n, m, 13), _rigid_body_pos=torch.zeros(n, m, 15, 3),
        _scene_ids=torch.full((n,), WAREHOUSE), _templates=torch.tensor([[5, 5, 3, 1]]).repeat(n, 1),
        _before_predecessor=torch.tensor([[-1, -1, -1, 2]]).repeat(n, 1),
        _warehouse_before_probability=.2, _warehouse_before_pairs=torch.zeros(n, dtype=torch.bool),
        _warehouse_door_passed=torch.zeros(n, m, dtype=torch.bool),
        _env_origins=torch.full((n, 3), 7.), _structure_shift=torch.full((n, 4, 2), 2.),
        _structure_yaw=torch.full((n, 4), yaw), _surface_half=half, _static_half=static,
        _warehouse={'wall_thickness': .4},
        _relation_cfg={'progress': {'sigma': 1.}, 'progress_reward_weight': .2}, extras={})


def move(task, x, y=0., z=1.):
    local = torch.zeros(task.num_envs, task.num_agents, 2)
    local[..., 0], local[..., 1] = x, y
    task._humanoid_root_states[:, :, :2] = scene.to_world(local, task._structure_yaw[:, WAREHOUSE],
        task._structure_shift[:, WAREHOUSE], task._env_origins)
    task._humanoid_root_states[:, :, 2] = z


def test_config_adds_door_reward_and_disables_reset_rsi():
    cfg, train = native_config(NAME), native_config(NAME, True)
    validate_experiment(NAME, cfg['env'], train)
    assert experiment_horizon(NAME) == 900
    restored = deepcopy(cfg)
    restored['experiment']['name'] = BASE
    assert restored['env']['sensorScene']['warehouse'].pop('leader_door_progress') is True
    assert restored['env']['stateInit'] == 'Default'
    start = restored['env']['sensorScene']['warehouse']['start_distribution']
    assert start['probabilities'] == [1., 0., 0.]
    restored['env']['stateInit'] = native_config(BASE)['env']['stateInit']
    start['probabilities'] = native_config(BASE)['env']['sensorScene']['warehouse']['start_distribution']['probabilities']
    assert restored == native_config(BASE)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    bad = deepcopy(cfg['env'])
    bad['sensorScene']['warehouse']['leader_door_progress'] = .2
    with pytest.raises(ValueError, match='must be boolean'):
        validate_experiment(NAME, bad, native_config(NAME, True))


def test_only_the_leader_of_the_two_push_one_before_scene_receives_reward():
    task = make_task(4)
    task._templates[1] = torch.tensor([3, 1, 3, 4])
    task._before_predecessor[1] = torch.tensor([-1, 0, -1, 2])
    task._warehouse_before_pairs[1] = True
    task._scene_ids[2] = ROOM
    task._before_predecessor[3] = -1
    move(task, -5.2)
    result = reward(task)
    expected = torch.zeros(4, 4)
    expected[0, 2] = .1
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('yaw', [0., 1.3])
def test_xy_progress_reuses_the_existing_scale_and_latches_after_passing(yaw):
    task = make_task(yaw=yaw)
    for distance in (3., 2., 1.):
        move(task, -4.2-distance, z=100.)
        torch.testing.assert_close(reward(task)[0, 2], torch.tensor(.2/(1+distance)))
    move(task, -4.1, z=-100.)
    torch.testing.assert_close(reward(task)[0, 2], torch.tensor(.2))
    assert task._warehouse_door_passed[0, 2]
    for x, y in ((3., 3.), (-9., 4.)):
        move(task, x, y)
        torch.testing.assert_close(reward(task)[0, 2], torch.tensor(.2))
        assert task.extras['warehouse_door_progress'][0, 2] == 1.


def test_passing_the_front_plane_outside_the_door_width_does_not_saturate():
    task = make_task()
    move(task, -4., 3.)
    result = reward(task)
    assert 0 < result[0, 2] < .2
    assert not task._warehouse_door_passed.any()


def test_leader_selection_follows_the_predecessor_binding():
    task = make_task()
    task._templates[0] = torch.tensor([1, 3, 5, 5])
    task._before_predecessor[0] = torch.tensor([1, -1, -1, -1])
    move(task, -5.2)
    torch.testing.assert_close(reward(task), torch.tensor([[0., .1, 0., 0.]]))
