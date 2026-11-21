from copy import deepcopy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.square_sensor import rotation_matrix
from env.tasks.multi_agent.square_sensor_scene import yaw_quat
from env.tasks.multi_agent.warehouse_close import held_people, risks, combine, diagnostics
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import WAREHOUSE, validate_sensor
from utils.wall_proximity import segment_box_distance, capsule_wall_gap, box_box_gap, proximity
from utils.warehouse_sensor import walls


def test_config_changes_only_wall_settings_and_name():
    cfg = native_config('warehouse_close')
    base = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs')
    train = native_config('warehouse_close', True)
    validate_experiment('warehouse_close', cfg['env'], deepcopy(train))
    cfg['experiment']['name'] = base['experiment']['name']
    warehouse = cfg['env']['sensorScene']['warehouse']
    assert warehouse.pop('wall_cpa') == 'box'
    assert warehouse.pop('wall_proximity') == dict(distance=.1, coefficient=.7, payload=True)
    assert cfg == base
    other = native_config(base['experiment']['name'], True)
    assert train['params']['config']['name'] == 'WarehouseClose'
    train['params']['config']['name'] = other['params']['config']['name']
    assert train == other


@pytest.mark.parametrize('key,value', [('distance', 0), ('distance', float('nan')), ('coefficient', -1), ('payload', 1)])
def test_invalid_option(key, value):
    env = native_config('warehouse_close')['env']
    env['sensorScene']['warehouse']['wall_proximity'][key] = value
    with pytest.raises(ValueError, match='wall_proximity'):
        validate_sensor(env)


def test_exact_segment_corner_and_crossing_distance():
    start = torch.tensor([[-2., 0., 0.], [2., 2., 0.], [2., 2., 0.], [1.05, 0., 0.]])
    end = torch.tensor([[2., 0., 0.], [3., 3., 0.], [2., -2., 0.], [1.05, 0., 0.]])
    expected = torch.tensor([0., 2.**.5, 1., .05])
    torch.testing.assert_close(segment_box_distance(start, end, torch.ones(4, 3)), expected)


def test_box_distance_rotation_containment_and_crossed_edges():
    identity = torch.eye(3)[None]
    zero = torch.zeros(1, 3)
    half = torch.ones(1, 3)
    torch.testing.assert_close(box_box_gap(zero, identity, half, torch.tensor([[2.05, 0., 0.]]), identity, half), torch.tensor([.05]))
    assert box_box_gap(zero, identity, half, zero, identity, half / 5).item() == 0
    rotated = rotation_matrix(yaw_quat(torch.tensor([torch.pi/4])))
    gap = box_box_gap(torch.tensor([[3., 0., 0.]]), rotated, half, zero, identity, half)
    torch.testing.assert_close(gap, torch.tensor([2. - 2.**.5]))
    horizontal, vertical = torch.tensor([[2., .1, .1]]), torch.tensor([[.1, 2., .1]])
    assert box_box_gap(zero, identity, horizontal, zero, identity, vertical).item() == 0


def make_task():
    center, half, _, _ = walls(torch.tensor([[4., 4.]]), .4, 2.5, torch.tensor([3.72]))
    task = SimpleNamespace(num_envs=1, num_agents=4, device='cpu', _agent_collision_coeff=.5,
        _warehouse={'wall_proximity': dict(distance=.1, coefficient=.7, payload=True)},
        _wall_contact_body_ids=[0], _wall_proximity_sums=torch.zeros(6))
    task._static_centers = torch.zeros(1, WAREHOUSE+1, 6, 3)
    task._static_half = torch.zeros_like(task._static_centers)
    task._static_centers[:, WAREHOUSE, 1:] = center
    task._static_half[:, WAREHOUSE, 1:] = half
    task._static_quat = torch.tensor([0., 0., 0., 1.]).expand(1, WAREHOUSE+1, 6, 4)
    # Same 5 cm surface clearance outside/inside, then an open-door passage and distant person.
    task._rigid_body_pos = torch.tensor([[[[-4.65, 3., 1.]], [[-3.75, 3., 1.]], [[-4.2, 0., 1.]], [[0., 0., 1.]]]])
    task._rigid_body_rot = torch.tensor([0., 0., 0., 1.]).expand(1, 4, 1, 4)
    task._wall_proximity_shapes = (torch.zeros(1, 2, 3), torch.tensor([.2]), torch.tensor([0]))
    task._agent_box_assignment = torch.arange(4)[None]
    task._box_states = torch.zeros(1, 4, 13)
    task._box_states[..., 6] = 1
    task._box_states[..., :3] = torch.tensor([-3.75, 3., 1.])
    task._box_size = torch.full((1, 4, 3), .5)
    graph = SimpleNamespace(edge_valid=torch.ones(1, 4, dtype=torch.bool),
        edge_relation=torch.tensor([[6, 6, 6, 11]]), edge_owner=torch.tensor([[0, 1, 2, 3]]))
    task.relation_runtime = SimpleNamespace(graph=graph, own_success=torch.tensor([[False, True, True, True]]))
    return task


def test_stationary_body_clearance_is_symmetric_and_door_is_open():
    task = make_task()
    task._warehouse['wall_proximity']['payload'] = False
    human, payload = risks(task, torch.tensor([0]))
    torch.testing.assert_close(human, torch.tensor([[.5, .5, 0., 0.]]), atol=3e-6, rtol=0)
    assert not payload.any()
    torch.testing.assert_close(proximity(torch.tensor([-.1, 0., .05, .1, .2]), .1), torch.tensor([1., 1., .5, 0., 0.]))


def test_current_holding_only_and_no_double_penalty():
    task = make_task()
    ids = torch.tensor([0])
    assert held_people(task, ids).tolist() == [[False, True, True, False]]
    human, payload = risks(task, ids)
    torch.testing.assert_close(payload, torch.tensor([[0., 1., 1., 0.]]))
    existing = torch.tensor([[.2, .4, 1.2, 1.4]])
    combined = combine(task, ids, existing) * task._agent_collision_coeff
    torch.testing.assert_close(combined, torch.tensor([[.35, .7, .7, .7]]), atol=3e-6, rtol=0)
    task.relation_runtime.own_success.zero_()
    assert not risks(task, ids)[1].any()
    result = {}
    diagnostics(task, result)
    assert result['warehouse/proximity/added_penalty'] > 0
    assert not task._wall_proximity_sums.any()


def test_capsule_middle_can_touch_when_endpoints_are_clear():
    # A long capsule crosses a thin wall; endpoint-only sampling would miss it.
    shapes = (torch.tensor([[[-2., 0., 0.], [2., 0., 0.]]]), torch.tensor([.1]), torch.tensor([0]))
    gap = capsule_wall_gap(torch.zeros(1, 1, 1, 3), torch.eye(3).expand(1, 1, 1, 3, 3), shapes,
        torch.tensor([[[.4, 0., 0.]]]), torch.eye(3).expand(1, 1, 3, 3), torch.tensor([[[.05, .2, .2]]]))
    torch.testing.assert_close(gap, torch.tensor([[-.1]]))
