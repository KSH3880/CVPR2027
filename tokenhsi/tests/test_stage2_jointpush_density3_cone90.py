from pathlib import Path

import pytest
import torch
import yaml

from utils.stage2_jointpush import asset_bank, checkpoint_contract, check_checkpoint, validate_env

BASE = 'stage2_jointpush_goal_saturation'
NAME = BASE + '_density3_cone90'


def config(folder, name):
    return yaml.safe_load(Path('tokenhsi/data/cfg', folder, name + '.yaml').read_text())


def test_only_requested_environment_and_transfer_changes():
    original = config('multi_agent', BASE)
    variant = config('multi_agent', NAME)
    validate_env(variant['env'])
    check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(variant['env'])}, variant['env'])
    with pytest.raises(ValueError):
        check_checkpoint({'stage2_jointpush_contract': checkpoint_contract(original['env'])}, variant['env'])
    push = variant['env']['jointPush']
    assert push['box'].pop('density') == 3.
    push['box']['mass'] = 20.
    assert push['scene']['goal_angle_degrees'] == 90.
    push['scene']['goal_angle_degrees'] = 360.
    assert push['rsi'].pop('reference_mass') == 20.
    assert variant['experiment']['transfer']['checkpoint'] == 'pth/Stage1JointPushDensity3Cone90Finetune_00011000.pth'
    variant['experiment'] = original['experiment']
    assert variant == original
    original = config('train/rlg', BASE)
    variant = config('train/rlg', NAME)
    variant['params']['config']['name'] = original['params']['config']['name']
    assert variant == original


def test_density_scales_with_actual_volume_for_all_group_sizes():
    original = config('multi_agent', BASE)['env']
    variant = config('multi_agent', NAME)['env']
    torch.manual_seed(42)
    old_sizes, old_masses = asset_bank(1024, original, 'cpu')
    torch.manual_seed(42)
    sizes, masses = asset_bank(1024, variant, 'cpu')
    torch.testing.assert_close(sizes, old_sizes, rtol=0, atol=0)
    torch.testing.assert_close(masses[:, :16], old_masses[:, :16], rtol=0, atol=0)
    torch.testing.assert_close(masses[:, 16:], 3 * sizes[:, 16:].prod(-1))
    torch.testing.assert_close(masses[:, 20:22], 2 * masses[:, 16:18])
    torch.testing.assert_close(masses[:, 22:24], masses[:, 16:17] * torch.tensor([3., 4.]))
    assert len(sizes[:, 16:20].flatten(0, 1).unique(dim=0)) == 167
