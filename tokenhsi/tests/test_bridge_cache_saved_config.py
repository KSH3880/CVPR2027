from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from utils.experiments import native_config, load_environment, validate_experiment
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint

NAME = 'stage2_sensor_900_bridge_before_fixed_layout_cache'
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('evaluation', [True, False])
def test_legacy_saved_sampler_loads_without_weakening_checkpoint_contract(tmp_path, evaluation):
    saved = native_config(NAME)
    bridge = saved['env']['sensorScene']['bridge']
    bridge['distance_curriculum']['mixture'] = [.2, .6, .2]
    bridge['layout_cache']['refresh_probability'] = .01
    (tmp_path/'nn').mkdir()
    checkpoint = tmp_path/'nn/model.pth'
    checkpoint.touch()
    (tmp_path/'relation_config.yaml').write_text(yaml.safe_dump(saved))
    loaded, path = load_environment(NAME, '', str(checkpoint), saved=True)
    assert path == tmp_path/'relation_config.yaml'
    assert loaded['env'] == saved['env']
    validate_experiment(NAME, loaded['env'], native_config(NAME, True), evaluation=evaluation)
    weights = dict(coupled_push_contract=checkpoint_contract(saved['env']))
    check_checkpoint(weights, loaded['env'], inference=evaluation)
    with pytest.raises(ValueError):
        check_checkpoint(weights, native_config(NAME)['env'], inference=evaluation)


@pytest.mark.parametrize('changed', ['reward', 'same_platform', 'entries', 'missing_cache', 'disabled_cache', 'cache_option'])
def test_sampler_compatibility_still_rejects_other_environment_changes(changed):
    env = deepcopy(native_config(NAME)['env'])
    if changed == 'reward':
        env['relationReward']['progress_reward_weight'] += .1
    elif changed == 'same_platform':
        env['sensorScene']['bridge']['before_training']['same_platform_probability'] = .3
    elif changed == 'entries':
        env['sensorScene']['bridge']['layout_cache']['entries_per_bucket'] = 3
    elif changed == 'missing_cache':
        del env['sensorScene']['bridge']['layout_cache']
    elif changed == 'disabled_cache':
        env['sensorScene']['bridge']['layout_cache'] = None
    else:
        env['sensorScene']['bridge']['layout_cache']['unrecognized_mode'] = True
    with pytest.raises(ValueError):
        validate_experiment(NAME, env, native_config(NAME, True), evaluation=True)


@pytest.mark.parametrize('wrapper', ['train', 'train_vnc'])
@pytest.mark.parametrize('resume,explicit', [(False, None), (True, None), (False, 'custom.yaml'), (True, 'saved.yaml')])
def test_cache_wrappers_select_native_for_fresh_and_saved_for_resume(tmp_path, wrapper, resume, explicit):
    directory = tmp_path/'tokenhsi/scripts/multi_agent'
    directory.mkdir(parents=True)
    name = NAME+'_'+wrapper+'.sh'
    shutil.copy2(ROOT/'tokenhsi/scripts/multi_agent'/name, directory/name)
    (tmp_path/'mps').mkdir()
    (tmp_path/'mps/shell.sh').write_text('mps_use() { :; }\n')
    (directory/'launch.sh').write_text('printf "%s" "${ENV_CONFIG-}"\n')
    binary = tmp_path/'bin'
    binary.mkdir()
    (binary/'nvidia-smi').write_text('#!/bin/sh\nprintf "5\\n"\n')
    (binary/'nvidia-smi').chmod(0o755)
    env = os.environ.copy()
    env.pop('ENV_CONFIG', None)
    env.pop('RESUME_CHECKPOINT', None)
    env['PATH'] = str(binary)+os.pathsep+env['PATH']
    if resume:
        env['RESUME_CHECKPOINT'] = 'model.pth'
    if explicit is not None:
        env['ENV_CONFIG'] = explicit
    result = subprocess.run(['/bin/bash', str(directory/name)], env=env, check=True, capture_output=True, text=True)
    expected = explicit if explicit is not None else ('' if resume else 'tokenhsi/data/cfg/multi_agent/'+NAME+'.yaml')
    assert result.stdout == expected
