from copy import deepcopy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent import warehouse_done
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import WAREHOUSE


NAME = 'stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox'
BASE = 'stage2_sensor_900_warehouse_fixedroles_v4'


def test_config_adds_only_leader_done_and_wall_box_cpa():
    cfg, train = native_config(NAME), native_config(NAME, True)
    validate_experiment(NAME, cfg['env'], train)
    restored = deepcopy(cfg)
    restored['experiment']['name'] = BASE
    warehouse = restored['env']['sensorScene']['warehouse']
    assert warehouse['cooperative'].pop('leader_done_probability') == .3
    assert warehouse.pop('wall_cpa') == 'box'
    env, warehouse = restored['env'], restored['env']['sensorScene']['warehouse']
    assert (env['stateInit'], warehouse['start_distribution']['probabilities']) == ('Default', [1., 0., 0.])
    env['stateInit'], warehouse['start_distribution']['probabilities'] = 'Random', [.5, .25, .25]
    assert env.pop('warehousePushRegionMatching') is True
    assert warehouse.pop('layout_cache') == dict(capacity=32, samples=2, refresh_probability=.25, warmup_resets=32)
    assert restored == native_config(BASE)
    base_train = native_config(BASE, True)
    train['params']['config']['name'] = base_train['params']['config']['name']
    assert train == base_train


@pytest.mark.parametrize('key,value', [('leader_done_probability', 1.1), ('leader_done_probability', -.1),
                                       ('before_pairs_probability', .2), ('pusher_probabilities', [0., 0., 1.])])
def test_leader_done_requires_the_fixed_role_warehouse(key, value):
    env = native_config(NAME)['env']
    env['sensorScene']['warehouse']['cooperative'][key] = value
    with pytest.raises(ValueError):
        validate_experiment(NAME, env, native_config(NAME, True))


def test_sampling_marks_only_warehouse_resets():
    torch.manual_seed(0)
    n = 20000
    task = SimpleNamespace(num_envs=n, device='cpu', _warehouse_done_probability=.3)
    warehouse_done.allocate(task)
    ids = torch.arange(n)
    warehouse = torch.rand(n) < .8
    warehouse_done.sample(task, ids, warehouse)
    flag = task._warehouse_leader_done
    assert not flag[~warehouse].any()
    assert abs(flag[warehouse].float().mean()-.3) < .015
    assert task._warehouse_done_samples.tolist() == [float((warehouse & ~flag).sum()), float(flag.sum())]


def make_place_task(n=64, seed=0):
    torch.manual_seed(seed)
    m, slots, hx, hy = 4, 8, 4., 4.
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    yaw, shift, origin = torch.rand(n)*6.28, torch.randn(n, 2), torch.randn(n, 3)*20
    world = lambda local: scene.to_world(local, yaw, shift, origin)
    pre = torch.tensor([-1, -1, -1, 2]).expand(n, m).clone()
    goal = torch.stack((-hx-.4-.5-torch.rand(n)*2.6, (2*torch.rand(n)-1)*(hy-.4)), -1)
    roots = torch.stack((torch.tensor([1., 1., -6., 0.]).expand(n, m), torch.tensor([-1., 1., 3., 0.]).expand(n, m)), -1).clone()
    roots[:, 3] = goal+torch.tensor([0., 2.])*torch.sign(-goal[:, 1:])   # the follower waits two metres away
    states = torch.zeros(n, m, 13)
    states[..., :2], states[..., 2] = world(roots), .9+cfg['floor_height']
    boxes = torch.zeros(n, slots, 13)
    boxes[..., :2] = world(torch.stack((torch.linspace(-3., 3., slots).expand(n, slots), torch.full((n, slots), 2.)), -1))
    boxes[..., 3:7] = scene.yaw_quat(yaw)[:, None]
    tar = torch.zeros(n, m, 3)
    tar[:, 2, :2], tar[:, 2, 2] = world(goal[:, None])[:, 0], origin[:, 2]+.25+cfg['floor_height']
    half = torch.zeros(n, WAREHOUSE+1, 2, 2)
    half[:, WAREHOUSE, 0] = torch.tensor([hx, hy])
    lifts = []

    def place_person(env, person, xy, heading, lift):
        task._humanoid_root_states[env, person, :2] = xy
        lifts.append(lift)

    task = SimpleNamespace(num_agents=m, num_physical_objects=slots, device='cpu',
        _warehouse_leader_done=torch.arange(n) % 2 == 0, _place_failed=torch.zeros(n, dtype=torch.bool),
        _before_predecessor=pre, _before_near_pairs=pre >= 0, _agent_box_assignment=torch.tensor([7, 7, 2, 3]).expand(n, m),
        _box_states=boxes, _box_size=torch.tensor([.5, .5, .5]).expand(n, slots, 3), _tar_pos=tar,
        _structure_yaw=yaw[:, None].expand(n, WAREHOUSE+1), _structure_shift=shift[:, None].expand(n, WAREHOUSE+1, 2),
        _env_origins=origin, _surface_half=half, _warehouse=cfg, _warehouse_coop=cfg['cooperative'],
        _humanoid_root_states=states, _active_physical=lambda ids: torch.ones(len(ids), slots, dtype=torch.bool),
        _place_person=place_person)
    to_local = lambda xy: scene.rotate_xy(xy-origin[:, None, :2]-shift[:, None], -yaw[:, None])
    return task, to_local, lifts


def test_leader_done_places_the_box_on_its_goal_and_the_leader_beside_it():
    task, to_local, lifts = make_place_task()
    before = (task._box_states.clone(), task._humanoid_root_states.clone(), task._before_near_pairs.clone())
    warehouse_done.place(task, torch.arange(64))
    done = task._warehouse_leader_done
    assert not task._place_failed.any()
    torch.testing.assert_close(task._box_states[done, 2, :3], task._tar_pos[done, 2])
    assert torch.equal(task._box_states[done][:, [0, 1, 3, 4, 5, 6, 7]], before[0][done][:, [0, 1, 3, 4, 5, 6, 7]])
    assert torch.equal(task._box_states[~done], before[0][~done])
    leader = to_local(task._humanoid_root_states[:, 2:3, :2])[:, 0]
    box = to_local(task._box_states[:, 2:3, :2])[:, 0]
    gap = ((leader-box).abs()-.25).clamp_min(0).norm(dim=-1)[done]
    assert ((gap >= .45-1e-5) & (gap <= 1.+1e-5)).all()
    assert (leader[done, 0] <= -4.-.4-.4+1e-5).all() and (leader[done, 0] >= -4.-9.+.4-1e-5).all()
    others = [0, 1, 3]
    assert torch.equal(task._humanoid_root_states[:, others], before[1][:, others])
    assert torch.equal(task._humanoid_root_states[~done], before[1][~done])
    assert not task._before_near_pairs[done].any() and torch.equal(task._before_near_pairs[~done], before[2][~done])
    assert all(not lift.any() for lift in lifts)


def test_failed_layouts_stay_failed_and_crowded_leaders_are_rejected():
    task, _, _ = make_place_task(8)
    task._place_failed[0] = True
    before = task._box_states.clone()
    task._humanoid_root_states[2, 3, :2] = task._tar_pos[2, 2, :2]   # the follower stands on the leader's only spot
    task._humanoid_root_states[2, 0, :2] = task._tar_pos[2, 2, :2]
    task._box_states[2, 0, :2] = task._tar_pos[2, 2, :2]+torch.tensor([.9, 0.])
    task._box_states[2, 1, :2] = task._tar_pos[2, 2, :2]-torch.tensor([.9, 0.])
    task._box_states[2, 3, :2] = task._tar_pos[2, 2, :2]+torch.tensor([0., .9])
    task._box_states[2, 4, :2] = task._tar_pos[2, 2, :2]-torch.tensor([0., .9])
    task._box_size = torch.tensor([1.2, 1.2, .5]).expand(8, 8, 3).clone()
    task._box_size[:, 2] = .5
    warehouse_done.place(task, torch.arange(8))
    assert torch.equal(task._box_states[0], before[0])
    assert task._place_failed[0] and task._place_failed[2]


def test_record_splits_main_and_leader_done_episodes():
    n = 4
    task = SimpleNamespace(num_envs=n, device='cpu', _scene_ids=torch.tensor([WAREHOUSE, WAREHOUSE, WAREHOUSE, 0]),
        _before_predecessor=torch.tensor([-1, -1, -1, 2]).expand(n, 4),
        _edge_ever_goal=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool))
    warehouse_done.allocate(task)
    task._warehouse_leader_done[:] = torch.tensor([False, True, True, True])
    warehouse_done.record(task, torch.tensor([True, True, True, True]))
    result = {}
    warehouse_done.diagnostics(task, result)
    assert result['warehouse/main/episodes'] == 1 and result['warehouse/leader_done/episodes'] == 2
    assert result['warehouse/main/leader_success'] == 0 and result['warehouse/main/follower_success'] == 0
    assert result['warehouse/leader_done/leader_success'] == 1 and result['warehouse/leader_done/follower_success'] == .5
    assert not task._warehouse_done_outcomes.any()
