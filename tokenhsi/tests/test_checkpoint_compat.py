"""Only names are migrated; historical tensors and physical contracts stay intact."""
from copy import deepcopy

import pytest
import yaml

from utils.experiments import NAMES, BEFORE_NAMES, native_config, load_environment
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata


@pytest.mark.parametrize('name', NAMES)
def test_old_snapshot_names_load_without_mutating_the_source(tmp_path, name):
    saved = native_config(name)
    saved['env']['relationReward']['stage1_variant'] = 'stage1_v2'
    saved['env']['sizeAwareRsi']['cacheDirectory'] = 'output/rsi_cache_stage1_v2'
    old = deepcopy(saved)
    path = tmp_path/'relation_config.yaml'
    path.write_text(yaml.safe_dump(saved))
    loaded, _ = load_environment(name, str(path))
    assert loaded['env'] == native_config(name)['env']
    assert saved == old


@pytest.mark.parametrize('name', BEFORE_NAMES)
def test_historical_checkpoint_contract_names_migrate_without_relaxing_rewards(name):
    env = native_config(name)['env']
    contract = deepcopy(checkpoint_contract(env))
    if 'size_aware_rsi' in contract:
        contract['size_aware_rsi']['cacheDirectory'] = 'output/rsi_cache_stage1_v2'
    before = deepcopy(contract)
    check_checkpoint({'before_contract': contract}, env)
    assert contract == before
    contract['before']['early_penalty'] = 1.
    with pytest.raises(ValueError):
        check_checkpoint({'before_contract': contract}, env)


def test_reward_metadata_migrates_only_the_name():
    expected = checkpoint_metadata(native_config(NAMES[0])['env']['relationReward'])
    old = deepcopy(expected)
    old['relation_reward_config']['stage1_variant'] = 'stage1_v2'
    check_checkpoint_metadata({'relation_metadata': old}, expected)
    old['relation_reward_config']['state_reward_weight'] = .9
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': old}, expected)


def test_local_checkpoint_alias_migrates_and_external_source_path_is_preserved():
    from utils.checkpoint_compat import checkpoint_path
    assert checkpoint_path('checkpoints/Stage1V2Fixed_00078000.pth') == 'checkpoints/Stage1_00078000.pth'
    external = '/external/run/Stage1V2Fixed_00078000.pth'
    assert checkpoint_path(external) == external
