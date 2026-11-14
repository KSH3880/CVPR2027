from copy import deepcopy

import pytest

from test_evaluation_protocol import args
from utils.evaluation_protocol import configure
from utils.experiments import NAMES, experiment_horizon, native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3'
NAME = BASE + '_ep900'


def test_fresh_training_only_changes_episode_length_and_names():
    cfg, base = native_config(NAME), native_config(BASE)
    train = native_config(NAME, True)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], deepcopy(train))
    assert cfg['env']['episodeLength'] == 900
    assert not train['params']['load_checkpoint']
    assert train['params']['config']['horizon_length'] == 32
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    cfg['env']['episodeLength'] = 1200
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


@pytest.mark.parametrize('name,length,wrong', [(NAME, 900, 1200), (BASE, 1200, 900)])
def test_training_and_formal_evaluation_use_selected_horizon(name, length, wrong):
    cfg = native_config(name)
    assert experiment_horizon(name) == length
    cfg['env']['episodeLength'] = wrong
    with pytest.raises(ValueError, match=f'requires {length} steps'):
        validate_experiment(name, cfg['env'], native_config(name, True))
    configure(args(), cfg, name)
    assert cfg['env']['episodeLength'] == length
    validate_experiment(name, cfg['env'], native_config(name, True))


def test_existing_experiment_horizons_are_preserved():
    for name in NAMES:
        if name not in (NAME, 'stage2_before_total', 'stage2_before_total_ft21k'):
            assert experiment_horizon(name) == (1200 if name.startswith('stage2_') else 900)
