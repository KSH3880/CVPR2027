"""Normalize historical names at the checkpoint boundary, without changing tensors."""
from copy import deepcopy


def reward_config(config):
    result = deepcopy(config)
    if result.get('stage1_variant') == 'stage1_v2':
        result['stage1_variant'] = 'stage1'
    return result


def rsi_config(config):
    result = deepcopy(config)
    if result.get('cacheDirectory') == 'output/rsi_cache_stage1_v2':
        result['cacheDirectory'] = 'output/rsi_cache_stage1'
    return result


def environment(env):
    result = deepcopy(env)
    if 'relationReward' in result:
        result['relationReward'] = reward_config(result['relationReward'])
    if 'sizeAwareRsi' in result:
        result['sizeAwareRsi'] = rsi_config(result['sizeAwareRsi'])
    return result


def before_contract(contract):
    result = deepcopy(contract)
    if result is not None and 'size_aware_rsi' in result:
        result['size_aware_rsi'] = rsi_config(result['size_aware_rsi'])
    return result


def checkpoint_path(path):
    if path == 'checkpoints/Stage1V2Fixed_00078000.pth':
        return 'checkpoints/Stage1_00078000.pth'
    return path
