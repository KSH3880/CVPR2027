from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before_carry_rsi import HumanoidMASensorBridgeBeforeCarryRsi
from env.tasks.multi_agent.reference_motion import ReferenceMotionMixin
from utils.bridge_before_carry_rsi import NAME, validate
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract
from utils.experiments import experiment_horizon, native_config, validate_experiment

BASE = 'stage2_sensor_900_bridge_before_fixed'


def test_config_differs_from_fixed_only_by_carry_rsi_and_hidebox():
    cfg = native_config(NAME)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    base = native_config(BASE)
    assert base['env']['stateInit'] == 'Default' and cfg['env']['stateInit'] == 'Random'
    base['env']['stateInit'] = 'Random'
    base['env']['sensorScene']['bridge'].pop('before_training')
    assert checkpoint_contract(cfg['env']) != checkpoint_contract(base['env'])
    with pytest.raises(ValueError):
        check_checkpoint(dict(coupled_push_contract=checkpoint_contract(base['env'])), cfg['env'], True)
    env = deepcopy(cfg['env'])
    assert env['sensorScene']['sensor'].pop('exclude_own_payload') is True
    env['sensorScene']['bridge'].pop('carry_boundary_rsi')
    assert env == base['env'] and cfg['experiment']['transfer'] == base['experiment']['transfer']
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


def test_validate_requires_spec_and_hidebox():
    env = native_config(NAME)['env']
    bad = deepcopy(env)
    bad['sensorScene']['sensor']['exclude_own_payload'] = False
    with pytest.raises(ValueError):
        validate(bad)
    bad = deepcopy(env)
    bad['sensorScene']['bridge']['carry_boundary_rsi']['episode_probability'] = .6
    with pytest.raises(ValueError):
        validate(bad)


def test_excluded_box_is_invisible_only_to_its_mask():
    centers = torch.tensor([[[0., 0., .5], [3., 0., 1.]]])
    quats = torch.tensor([[[0., 0., 0., 1.]]*2])
    halves = torch.full((1, 2, 3), .5)
    xy = torch.tensor([[[0., 0.], [3., 0.]]])
    floor = torch.zeros(1)
    plain = sensor.vertical_height(xy, centers, quats, halves, floor)
    torch.testing.assert_close(plain, sensor.vertical_height(
        xy, centers, quats, halves, floor, exclude=torch.zeros(1, 2, dtype=torch.bool)))
    hidden = sensor.vertical_height(xy, centers, quats, halves, floor, exclude=torch.tensor([[True, False]]))
    torch.testing.assert_close(hidden, torch.tensor([[0., 1.5]]))


def test_carry_history_uses_carry_motion(monkeypatch):
    calls = []
    monkeypatch.setattr(ReferenceMotionMixin, '_init_amp_obs_ref', lambda self, slots, mids, times, skill: calls.append(('mixin', skill)))
    monkeypatch.setattr(HumanoidMACoupledPush, '_init_amp_obs_ref', lambda self, slots, mids, times, skill: calls.append(('push', skill)))
    task = object.__new__(HumanoidMASensorBridgeBeforeCarryRsi)
    slots = (torch.zeros(1, dtype=torch.long),)*2
    task._init_amp_obs_ref(slots, torch.zeros(1), torch.zeros(1), 'carryWith')
    task._init_amp_obs_ref(slots, torch.zeros(1), torch.zeros(1), 'loco')
    assert calls == [('mixin', 'carryWith'), ('push', 'loco')]
