from copy import deepcopy
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from env.tasks.multi_agent.bridge_surface import write_asset, load_asset
from env.tasks.multi_agent.sensor_stage2_observation import SensorStage2Observation
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract
from utils.experiments import native_config, validate_experiment, experiment_horizon, load_environment
from utils.sensor_stage2_spec import validate_sensor


BASE = 'stage2_sensor_900_cliff_bridge'
NAME = BASE+'_surface'


def test_config_and_checkpoint_isolation(tmp_path):
    cfg, old = native_config(NAME), native_config(BASE)
    current = deepcopy(cfg)
    current['env']['sensorScene']['bridge'].pop('top_surface')
    current['env']['sensorScene']['bridge'].pop('humanoid_height_reference')
    current['experiment']['name'] = BASE
    assert current == old
    train, old_train = native_config(NAME, True), native_config(BASE, True)
    train['params']['config']['name'] = old_train['params']['config']['name']
    assert train == old_train
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    weights = dict(coupled_push_contract=checkpoint_contract(cfg['env']))
    for inference in (False, True):
        check_checkpoint(weights, cfg['env'], inference=inference)
        with pytest.raises(ValueError):
            check_checkpoint(weights, old['env'], inference=inference)
        with pytest.raises(ValueError):
            check_checkpoint(dict(coupled_push_contract=checkpoint_contract(old['env'])), cfg['env'], inference=inference)
    import yaml
    (tmp_path/'relation_config.yaml').write_text(yaml.safe_dump(cfg))
    loaded, _ = load_environment(NAME, None, tmp_path/'policy.pth', saved=True)
    assert loaded['env'] == cfg['env']


@pytest.mark.parametrize('option,value', [('humanoid_height_reference', 'root'),
    ('top_surface', dict(thickness=.01, friction=1.)), ('top_surface', dict(thickness=.005, friction=.4))])
def test_invalid_surface_settings(option, value):
    env = native_config(NAME)['env']
    env['sensorScene']['bridge'][option] = value
    with pytest.raises(ValueError):
        validate_sensor(env)


class CaptureNodes:
    def _compute_clean_humanoid_nodes(self, body_pos, *args):
        return body_pos


class Observation(SensorStage2Observation, CaptureNodes):
    pass


def fixture():
    task = Observation()
    task.device, task.num_envs = 'cpu', 4
    task._scene_cfg = native_config(NAME)['env']['sensorScene']
    task._scene_ids = torch.tensor([1, 0, 1, 2])
    task._start_surface = torch.tensor([[1, 2, 1, 0]]).expand(4, -1)
    task._surface_height = torch.tensor([1.51, 1.61, 1.71, .1])[:, None].expand(-1, 3)
    task._env_origins = torch.tensor([[0., 0., 2.], [0., 0., 0.], [0., 0., 5.], [0., 0., 0.]])
    task._terrain_ids, task._bridge_entry_chamfer = None, False
    task._bridge_slot = torch.zeros(4, dtype=torch.long)
    task._static_centers = torch.zeros(4, 3, 6, 3)
    task._static_centers[..., 2] = -1000.
    task._static_half = torch.zeros_like(task._static_centers)
    task._static_quat = torch.zeros(4, 3, 6, 4)
    task._static_quat[..., 3] = 1.
    for env in range(4):
        height = task._surface_height[env, 0]
        for scenario in range(3):
            for slot, x in enumerate([0., 6.2]):
                task._static_centers[env, scenario, slot] = torch.tensor([x, 0., task._env_origins[env, 2]+height/2])
                task._static_half[env, scenario, slot] = torch.tensor([2., 2., height/2])
    task._box_size = torch.stack([torch.tensor([2., 2., task._surface_height[e, 0]-.01]) for e in range(4)])[:, None]
    task._box_states = torch.zeros(4, 1, 13)
    task._box_states[:, 0, 0], task._box_states[:, 0, 6] = 3.1, 1.
    task._box_states[:, 0, 2] = task._env_origins[:, 2]+task._box_size[:, 0, 2]/2
    return task


@pytest.mark.parametrize('subset', [None, [3, 2, 0]])
def test_height_is_continuous_across_gap_and_partial_reset(subset):
    task = fixture()
    ids = torch.arange(4) if subset is None else torch.tensor(subset)
    task._terrain_ids = None if subset is None else ids
    body = torch.zeros(len(ids), 4, 15, 3)
    body[..., 2] = (task._env_origins[ids, 2, None]+task._surface_height[ids, 1, None]+.92)[..., None]
    body[:, 3, :, 2] = task._env_origins[ids, 2, None]+.92
    expected = []
    for x in [1.8, 2.05, 2.3]:
        body[..., 0] = x
        body[:, 3, :, 0] = 10.
        before = body.clone()
        ground = task._terrain_height(body[..., 0, :2], task._terrain_ids)
        result = task._compute_clean_humanoid_nodes(body, None, None, None, task._env_origins[ids])
        bridge = task._scene_ids[ids] == 1
        torch.testing.assert_close(result[bridge, :, 0, 2], torch.full_like(result[bridge, :, 0, 2], .92))
        torch.testing.assert_close(result[~bridge, ..., 2], body[~bridge, ..., 2]-ground[~bridge, :, None])
        assert torch.equal(body, before)
        assert torch.equal(ground, task._terrain_height(body[..., 0, :2], task._terrain_ids))
        expected.append(result[bridge, :, 0, 2])
    assert torch.equal(expected[0], expected[1]) and torch.equal(expected[1], expected[2])
    body[..., 2] -= .3
    moved = task._compute_clean_humanoid_nodes(body, None, None, None, task._env_origins[ids])
    torch.testing.assert_close(moved[bridge, :, 0, 2], torch.full_like(moved[bridge, :, 0, 2], .62))
    task._scene_cfg['bridge'].pop('humanoid_height_reference')
    legacy = task._compute_clean_humanoid_nodes(body, None, None, None, task._env_origins[ids])
    torch.testing.assert_close(legacy[..., 2], body[..., 2]-task._terrain_height(body[..., 0, :2], task._terrain_ids)[..., None], atol=0, rtol=0)


@pytest.mark.parametrize('span', [1, 2, 3])
@pytest.mark.parametrize('height', [1.45, 1.7])
@pytest.mark.parametrize('controlled,density', [(False, 3.), (True, 500.)])
def test_cap_geometry_mass_inertia_and_cache(tmp_path, span, height, controlled, density):
    size = [1.7*span, 2.2, height]
    path = write_asset(tmp_path, size, density, .005, controlled)
    link = ET.parse(path).getroot().find('link')
    assert len(link.findall('visual')) == 1
    assert [float(x) for x in link.find('visual/geometry/box').get('size').split()] == size
    shapes = link.findall('collision')
    assert len(shapes) == 2
    bounds = []
    for shape in shapes:
        dims = [float(x) for x in shape.find('geometry/box').get('size').split()]
        z = float(shape.find('origin').get('xyz').split()[2])
        assert dims[:2] == size[:2]
        bounds.append([z-dims[2]/2, z+dims[2]/2])
    assert bounds[0] == pytest.approx([-height/2, height/2-.005])
    assert bounds[1] == pytest.approx([height/2-.005, height/2])
    mass = size[0]*size[1]*height*density
    assert float(link.find('inertial/mass').get('value')) == pytest.approx(mass)
    inertia = link.find('inertial/inertia')
    for key, value in zip(['ixx','iyy','izz'], [mass*(size[1]**2+height**2)/12,
            mass*(size[0]**2+height**2)/12, mass*(size[0]**2+size[1]**2)/12]):
        assert float(inertia.get(key)) == pytest.approx(value)
    content = path.read_bytes()
    other = write_asset(tmp_path, size, density+1, .005, not controlled)
    assert path != other and path.read_bytes() == content


def test_load_asset_checks_collision_topology(tmp_path):
    gym = SimpleNamespace(load_asset=lambda *args: object(), get_asset_rigid_body_count=lambda asset: 1,
        get_asset_rigid_shape_count=lambda asset: 2)
    assert load_asset(gym, None, tmp_path, [1.7, 2.2, 1.5], SimpleNamespace(density=3.), .005)
    gym.get_asset_rigid_shape_count = lambda asset: 1
    with pytest.raises(RuntimeError):
        load_asset(gym, None, tmp_path, [1.7, 2.2, 1.5], SimpleNamespace(density=3.), .005)


def test_formal_evaluation_uses_stage2_push_route(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    path = Path(__file__).resolve().parents[1]/'evaluate.py'
    spec = importlib.util.spec_from_file_location('surface_evaluate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path/'policy.pth'
    checkpoint.write_bytes(b'command routing check; not a model')
    output = tmp_path/'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', NAME, str(checkpoint), '--output', str(output),
        '--envs', '1', '--seeds', '0'])
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kwargs: (commands.append(cmd) or SimpleNamespace(returncode=0)))
    module.main()
    assert commands
    for cmd in commands:
        assert cmd[cmd.index('--task')+1] == 'HumanoidMACoupledPush'
        assert cmd[cmd.index('--motion_file')+1] == 'tokenhsi/data/dataset_stage1_jointpush.yaml'
        assert cmd[cmd.index('--cfg_train')+1].endswith(NAME+'.yaml')
    import json
    assert 'PUSH' in json.loads((output/'manifest.json').read_text())['tasks']
