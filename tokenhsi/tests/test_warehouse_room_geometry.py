from types import SimpleNamespace, MethodType

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.sensor_stage2_geometry import SensorStage2Geometry
from env.tasks.multi_agent.warehouse_coop import build_geometry, FIELDS
from utils.experiments import native_config
from utils.sensor_stage2_spec import ROOM


@pytest.mark.parametrize('seed', [0, 42, 984])
def test_warehouse_variants_preserve_room_dimensions_and_collision_walls(seed):
    torch.manual_seed(seed)
    spec = native_config('stage2_sensor_900_warehouse_coop_cliffroom')['env']['sensorScene']
    task = SimpleNamespace(num_envs=32, device='cpu', _scene_cfg=spec, _warehouse=spec['warehouse'],
        _warehouse_coop=spec['warehouse']['cooperative'], _bridge_training_init=spec['bridge']['start_distribution'],
        _scene_spacing=64., _geometry_bridge_slot=0, _bridge_bank=0)
    task._box_size = torch.tensor([1.8, 2., 1.5]).expand(32, 3, 3).clone()
    task._box_size[:, :, 0] *= torch.arange(1, 4)
    task._surface_halves = MethodType(SensorStage2Geometry._surface_halves, task)
    task._build_scene_geometry = MethodType(SensorStage2Geometry._build_scene_geometry, task)
    task._build_scene_geometry()
    fields = FIELDS+('_bridge_gap', '_bridge_far', '_bridge_half_a', '_bridge_half_b')
    original = {key: getattr(task, key).clone() for key in fields}
    room_half = task._room_half.clone()
    wall_local = tuple(value.clone() for value in task._wall_local)
    task._bridge_geometry = {key: value[:, None].expand(-1, 3, *value.shape[1:]).clone()
        for key, value in original.items()}

    build_geometry(task, fields)

    assert torch.equal(task._room_half, room_half)
    assert torch.equal(task._room_half, task._surface_half[:, ROOM, 0])
    assert all(torch.equal(a, b) for a, b in zip(task._wall_local, wall_local))
    assert all(torch.equal(getattr(task, key), value) for key, value in original.items())
    cfg = spec['room']
    _, _, starts, ends = scene.room_walls(task._surface_half[:, ROOM, 0],
        cfg['wall_thickness'], cfg['wall_height'], cfg['passage_width'])
    for actual, local in zip(task._wall_local, (starts, ends)):
        expected = scene.rotate_xy(local, task._structure_yaw[:, ROOM, None])+task._structure_shift[:, ROOM, None]
        torch.testing.assert_close(actual, expected)
    assert task._warehouse_geometry['_static_half'].shape[1] == 6
