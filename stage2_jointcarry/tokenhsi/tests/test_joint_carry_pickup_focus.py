from copy import deepcopy

import pytest
import torch

from utils.joint_carry_amp import check_ablation_checkpoint, validate_ablation
from utils.joint_carry_rsi import PICKUP_FRAME_SAMPLING, load_paired_snapshots, pickup_frame_weights
from utils.joint_carry_spec import validate_env
from test_joint_carry_amp_ablation import configs


BASE = 'locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding'
FOCUS = 'locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding'


def test_only_joint_pickup_sampling_changes():
    env, train = configs(FOCUS)
    base, base_train = configs(BASE)
    validate_env(env)
    validate_ablation(env, train)
    restored = deepcopy(env)
    assert restored['jointCarryRsi'].pop('pickupFrameSampling') == PICKUP_FRAME_SAMPLING
    assert restored == base
    restored = deepcopy(train)
    restored['params']['config']['name'] = base_train['params']['config']['name']
    assert restored == base_train
    saved = dict(joint_carry_pickup_sampling=PICKUP_FRAME_SAMPLING,
                 joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
                 joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
                 joint_carry_ablation=env['jointCarryAblation'])
    check_ablation_checkpoint(saved, env)
    with pytest.raises(ValueError, match='pickup frame sampling'):
        check_ablation_checkpoint(saved, base)
    saved.pop('joint_carry_pickup_sampling')
    check_ablation_checkpoint(saved, base)
    with pytest.raises(ValueError, match='pickup frame sampling'):
        check_ablation_checkpoint(saved, env)


def test_actual_frames_mixture_and_all_frame_support():
    env, _ = configs(FOCUS)
    rsi = env['jointCarryRsi']
    frames = load_paired_snapshots(rsi['directory'], 'cpu', True,
                                  rsi['dataset_sha256'])['pickUp']['source_frame']
    assert frames.tolist() == list(range(100))
    weights = pickup_frame_weights(frames, rsi['pickupFrameSampling'])
    assert (weights > 0).all()
    groups = [(frames >= 25) & (frames <= 39), (frames >= 55) & (frames <= 80)]
    groups.append(~(groups[0] | groups[1]))
    expected = torch.tensor([.395, .428, .177])
    torch.testing.assert_close(torch.stack([weights[g].sum() for g in groups]), expected)
    torch.manual_seed(42)
    rows = torch.multinomial(weights, 100000, replacement=True)
    observed = torch.stack([g[rows].float().mean() for g in groups])
    torch.testing.assert_close(observed, expected, atol=.005, rtol=0)
    assert rows.unique().numel() == 100
    order = torch.randperm(100)
    torch.testing.assert_close(pickup_frame_weights(frames[order], rsi['pickupFrameSampling']), weights[order])
    assert pickup_frame_weights(frames, None) is None


def test_reject_incomplete_data_and_wrong_contract():
    with pytest.raises(ValueError, match='every frame'):
        pickup_frame_weights(torch.arange(35, 100), PICKUP_FRAME_SAMPLING)
    env, _ = configs(FOCUS)
    env['jointCarryRsi']['pickupFrameSampling']['probabilities'] = [.7, .0, .3]
    with pytest.raises(ValueError, match='sampling contract'):
        validate_env(env)
