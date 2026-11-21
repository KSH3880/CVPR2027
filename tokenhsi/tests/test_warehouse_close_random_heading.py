from copy import deepcopy

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent.warehouse_sensor import randomize_before_heading
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import WAREHOUSE, validate_sensor
from utils.stage2_spec import before_members
from utils.torch_utils import calc_heading

NAME = 'warehouse_close_random_heading'


def test_config_only_changes_heading_and_names():
    cfg, base = native_config(NAME), native_config('warehouse_close')
    train, old_train = native_config(NAME, True), native_config('warehouse_close', True)
    validate_experiment(NAME, cfg['env'], deepcopy(train))
    assert cfg['env']['sensorScene']['warehouse'].pop('before_heading') == 'uniform'
    cfg['experiment']['name'] = base['experiment']['name']
    assert cfg == base
    train['params']['config']['name'] = old_train['params']['config']['name']
    assert train == old_train


@pytest.mark.parametrize('mode,state', [('typo', 'Default'), ('uniform', 'Random')])
def test_invalid_heading_configuration(mode, state):
    env = native_config(NAME)['env']
    env['stateInit'] = state
    env['sensorScene']['warehouse']['before_heading'] = mode
    with pytest.raises(ValueError, match='heading'):
        validate_sensor(env)


@pytest.mark.parametrize('follower_task', [1, 2, 3, 4])
def test_partial_reset_preserves_positions_pushers_and_pose(follower_task):
    task = SensorStage2Placement()
    task.num_agents, task.num_bodies, task.device = 4, 2, 'cpu'
    task._scene_ids = torch.tensor([WAREHOUSE, WAREHOUSE, WAREHOUSE, 0])
    task._before_predecessor = torch.tensor([[-1, -1, -1, 2], [-1, 0, -1, 2],
                                           [-1, -1, -1, 2], [-1, -1, -1, 2]])
    task._templates = torch.tensor([[5, 5, 4, follower_task], [4, follower_task, 4, follower_task],
                                    [5, 5, 4, follower_task], [5, 5, 4, follower_task]])
    root = torch.zeros(4, 4, 13)
    root[..., :3] = torch.arange(48).reshape(4, 4, 3)
    root[..., 6] = 1.
    root[..., 7] = 1.
    task._humanoid_root_states = root.clone()
    body = root[:, :, None].expand(-1, -1, 2, -1).clone()
    body[:, :, 1, 0] += .3
    task._kinematic_humanoid_rigid_body_states = body.clone()
    task._before_heading_generator = torch.Generator().manual_seed(123)
    rng = torch.get_rng_state().clone()
    ids = torch.tensor([0, 1, 3])
    randomize_before_heading(task, ids)
    changed = before_members(task._before_predecessor)
    changed[2:] = False
    assert torch.equal(task._humanoid_root_states[~changed], root[~changed])
    assert torch.equal(task._kinematic_humanoid_rigid_body_states[~changed], body[~changed])
    assert torch.equal(task._humanoid_root_states[..., :3], root[..., :3])
    offset = task._kinematic_humanoid_rigid_body_states[..., :3] - root[:, :, None, :3]
    torch.testing.assert_close(offset.norm(dim=-1), (body[..., :3] - root[:, :, None, :3]).norm(dim=-1))
    assert torch.equal(torch.get_rng_state(), rng)
    first = calc_heading(task._humanoid_root_states[..., 3:7].reshape(-1, 4)).reshape(4, 4).clone()
    randomize_before_heading(task, ids)
    second = calc_heading(task._humanoid_root_states[..., 3:7].reshape(-1, 4)).reshape(4, 4)
    assert ((first - second).abs()[changed] > 1e-3).all()


def test_independent_uniform_headings_and_reproducibility():
    from types import SimpleNamespace
    n = 8192
    task = SimpleNamespace(num_agents=4, device='cpu', _scene_ids=torch.full((n,), WAREHOUSE),
        _before_predecessor=torch.tensor([[-1, -1, -1, 2]]).expand(n, -1),
        _templates=torch.tensor([[5, 5, 4, 1]]).expand(n, -1),
        _humanoid_root_states=torch.zeros(n, 4, 13),
        _before_heading_generator=torch.Generator().manual_seed(19))
    values = {}
    task._place_person = lambda ids, person, xy, yaw, lift: values.update({person: yaw.clone()})
    randomize_before_heading(task, torch.arange(n))
    assert set(values) == {2, 3}
    for yaw in values.values():
        bins = torch.bincount((yaw / (torch.pi / 2)).long(), minlength=4)
        assert (bins.float() / n - .25).abs().max() < .025
    assert ((values[2] - values[3]).cos().mean()).abs() < .03
    saved = {k: v.clone() for k, v in values.items()}
    task._before_heading_generator.manual_seed(19)
    randomize_before_heading(task, torch.arange(n))
    assert all(torch.equal(saved[k], values[k]) for k in saved)
