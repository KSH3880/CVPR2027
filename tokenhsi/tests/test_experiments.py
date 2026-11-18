"""Retained profiles, saved physics and real Stage1 checkpoint compatibility."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from before_test_utils import build
from utils.experiments import (ROOT, NAMES, FIXED, BEFORE_NAMES, native_config,
                               load_environment, validate_experiment)
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.unified_training import validate_unified_env
from learning.multi_agent.before_network import transfer_stage1


@pytest.mark.parametrize('name', NAMES)
def test_profiles_validate_and_native_sources_are_not_mutated(name):
    cfg = native_config(name)
    before = deepcopy(cfg)
    train = native_config(name, training=True)
    validate_unified_env(cfg['env'])
    validate_experiment(name, cfg['env'], train)
    assert cfg == before
    assert cfg['env']['numEnvs'] == (2048 if name in (FIXED, 'stage1_push') else 1024)


@pytest.mark.parametrize('source', BEFORE_NAMES)
@pytest.mark.parametrize('target', BEFORE_NAMES)
def test_experiment_and_checkpoint_cross_use_is_rejected(source, target):
    env = native_config(source)['env']
    weights = {'before_contract': checkpoint_contract(env)}
    if source == target:
        validate_experiment(target, env, native_config(target, training=True))
        check_checkpoint(weights, env)
    else:
        with pytest.raises(ValueError):
            validate_experiment(target, env, native_config(target, training=True))
        with pytest.raises(ValueError):
            check_checkpoint(weights, native_config(target)['env'])


@pytest.mark.parametrize('name', NAMES)
def test_saved_env_and_sim_are_authoritative_and_explicit_yaml_wins(tmp_path, name):
    run = tmp_path/'run'
    (run/'nn').mkdir(parents=True)
    checkpoint = run/'nn/model.pth'
    checkpoint.touch()
    saved = native_config(name)
    saved['env']['episodeLength'] = 777
    saved['sim']['physx']['num_position_iterations'] = 9
    (run/'relation_config.yaml').write_text(yaml.safe_dump(saved))
    loaded, path = load_environment(name, '', str(checkpoint), saved=True)
    assert loaded['env']['episodeLength'] == 777
    assert loaded['sim'] == saved['sim']
    assert path == run/'relation_config.yaml'
    explicit = ROOT/f'tokenhsi/data/cfg/multi_agent/{name}.yaml'
    loaded, path = load_environment(name, str(explicit), str(checkpoint), saved=True)
    assert loaded['sim'] == native_config(name)['sim']
    assert path == explicit


def test_missing_saved_config_does_not_fall_back_to_current_yaml(tmp_path):
    with pytest.raises(ValueError, match='config missing'):
        load_environment(FIXED, '', str(tmp_path/'nn/model.pth'), saved=True)


@pytest.mark.parametrize('name', BEFORE_NAMES)
def test_actual_78000_checkpoint_transfers_without_discarding_tensors(name):
    path = ROOT/'checkpoints/Stage1_00078000.pth'
    if not path.is_file():
        pytest.skip('Connect the Stage1 78000 checkpoint under checkpoints/')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 78000
    model = torch.nn.Module()
    model.a2c_network = build(name)
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 78000
    assert set(report['copied']) == set(checkpoint['model'])
    for key, value in checkpoint['model'].items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.a2c_network.actor_encoder.parameters())
    assert all(p.requires_grad for p in model.a2c_network.action_head.parameters())
