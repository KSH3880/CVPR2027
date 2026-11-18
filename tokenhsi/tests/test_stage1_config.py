"""Physics opt-in isolation and saved-config restoration contracts."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from utils.stage1_config import MODE, resolve_config, uses_fixed_platforms

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return yaml.safe_load((ROOT / 'data/cfg/multi_agent' / (name + '.yaml')).read_text())




@pytest.mark.parametrize('mutation', ['unsupported', 'unknown_mode', 'height_disabled', 'no_platforms'])
def test_reject_invalid_physics_selection(mutation):
    env = config('stage1')['env']
    if mutation == 'unsupported':
        env['relationReward']['stage1_variant'] = 'unsupported'
    elif mutation == 'unknown_mode':
        env['box']['reset']['platformPhysics'] = 'typo'
    elif mutation == 'height_disabled':
        env['box']['reset']['randomHeight'] = False
    else:
        del env['box']['reset']['platformMode']
    with pytest.raises(ValueError, match='fixed_warmup_v1'):
        uses_fixed_platforms(env)


def test_restore_legacy_snapshot_preserves_saved_env_and_adds_native_sim():
    native = config('stage1')
    source = {'env': config('stage1')['env'], 'train': {'metadata': 'ignored'}}
    source['env']['box']['reset'].pop('platformPhysics')
    source['env']['episodeLength'] = 789
    before = deepcopy(source)
    resolved = resolve_config(source, native)
    assert resolved['sim'] == native['sim']
    assert resolved['sim']['physx']['rest_offset'] == 0
    assert resolved['env']['box']['reset'].pop('platformPhysics') == MODE
    assert resolved['env'] == source['env'] and source == before


def test_preserve_complete_saved_sim_settings():
    native = config('stage1')
    source = deepcopy(native)
    source['sim']['physx']['num_position_iterations'] = 9
    assert resolve_config(source, native) == source


@pytest.mark.parametrize('mutation', ['rest_offset', 'partial_sim', 'null_sim', 'unsupported'])
def test_do_not_silently_accept_incompatible_saved_config(mutation):
    native = config('stage1')
    source = deepcopy(native)
    if mutation == 'rest_offset':
        source['sim']['physx']['rest_offset'] = .001
    elif mutation == 'partial_sim':
        del source['sim']['physx']['max_depenetration_velocity']
    elif mutation == 'null_sim':
        source['sim'] = None
    else:
        source['env']['relationReward']['stage1_variant'] = 'unsupported'
    with pytest.raises(ValueError):
        resolve_config(source, native)
