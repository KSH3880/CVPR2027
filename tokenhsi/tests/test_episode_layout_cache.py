from copy import deepcopy

import pytest
import torch

from utils.episode_layout_cache import EpisodeLayoutCache, buckets
from utils.experiments import native_config, validate_experiment


def test_partial_restore_keeps_other_environments_and_types():
    bank = EpisodeLayoutCache(4, 2)
    ids = torch.tensor([0, 2])
    state = dict(pose=torch.arange(12.).reshape(4, 3), graph=torch.arange(8).reshape(4, 2),
                 valid=torch.tensor([True, False, True, False]))
    expected = {k: v.clone() for k, v in state.items()}
    bank.store(0, ids, torch.zeros(2), state)
    state['pose'][ids] = -100
    state['graph'][ids] = -1
    state['valid'][ids] = False
    slots = bank.lookup(0, ids, torch.zeros(2))
    bank.restore(0, ids, slots, state)
    for name in state:
        torch.testing.assert_close(state[name], expected[name], rtol=0, atol=0)


def test_distance_change_cannot_reuse_previous_curriculum():
    bank = EpisodeLayoutCache(2, 2)
    ids = torch.arange(2)
    state = dict(pose=torch.ones(2, 3))
    bank.store(1, ids, torch.full((2,), .1), state)
    assert bank.lookup(1, ids, torch.full((2,), .1)).ge(0).all()
    assert bank.lookup(1, ids, torch.full((2,), .1005)).eq(-1).all()
    assert bank.lookup(2, ids, torch.full((2,), .1)).eq(-1).all()


def test_multiple_entries_are_sampled_and_ring_replacement_is_bounded():
    bank = EpisodeLayoutCache(3, 2)
    ids = torch.arange(3)
    for value in (1., 2.):
        bank.store(0, ids, torch.zeros(3), dict(pose=torch.full((3, 1), value)))
    picked = torch.stack([bank.lookup(0, ids, torch.zeros(3)) for _ in range(50)])
    assert picked.eq(0).any() and picked.eq(1).any()
    bank.store(0, ids, torch.ones(3), dict(pose=torch.full((3, 1), 3.)))
    assert bank.lookup(0, ids, torch.ones(3)).eq(0).all()
    assert bank.lookup(0, ids, torch.zeros(3)).eq(1).all()


def test_buckets_separate_start_task_and_distance():
    mode = torch.arange(3).repeat_interleave(9)
    family = torch.tensor([1, 2, 4]).repeat_interleave(3).repeat(3)
    alpha = torch.tensor([0., .5, 1.]).repeat(9)
    assert buckets(mode, family, alpha).unique().numel() == 27


def test_final_distance_buckets_keep_both_platform_layouts_separate():
    mode = torch.arange(3).repeat_interleave(3).repeat(2)
    family = torch.tensor([1, 2, 4]).repeat(6)
    same = torch.arange(2).repeat_interleave(9).bool()
    alpha = torch.ones(18)
    kinds = buckets(mode, family, alpha, same, final_only=True)
    assert kinds.tolist() == list(range(18))
    bank = EpisodeLayoutCache(18, 2, 18)
    ids = torch.arange(18)
    state = dict(pose=torch.arange(54.).reshape(18, 3))
    expected = state['pose'].clone()
    for _ in range(2):
        bank.store(kinds, ids, alpha, state)
    state['pose'].zero_()
    bank.restore(kinds, ids, bank.lookup(kinds, ids, alpha), state)
    torch.testing.assert_close(state['pose'], expected, rtol=0, atol=0)


@pytest.mark.parametrize('update', [0, 501, 1400, 2500])
def test_native_cache_uses_final_distance_at_every_update(update):
    from utils.compact_bridge import sample_distance
    spec = native_config('stage2_sensor_900_bridge_before_fixed_layout_cache')['env']['sensorScene']['bridge']['distance_curriculum']
    assert sample_distance(100, update, spec, 'cpu').eq(1).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mixed_buckets_restore_in_one_batch(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    bank = EpisodeLayoutCache(4, 2)
    ids = torch.tensor([0, 1, 3], device=device)
    kinds = torch.tensor([0, 5, 26], device=device)
    alpha = torch.tensor([0., .25, 1.], device=device)
    state = dict(pose=torch.arange(12., device=device).reshape(4, 3))
    expected = state['pose'].clone()
    bank.store(kinds, ids, alpha, state)
    state['pose'][ids] = -1
    slots = bank.lookup(kinds, ids, alpha)
    assert slots.ge(0).all()
    bank.restore(kinds, ids, slots, state)
    torch.testing.assert_close(state['pose'], expected, rtol=0, atol=0)
    assert bank.lookup((kinds+1) % 27, ids, alpha).eq(-1).all()


def test_new_experiment_preserves_original_training_and_reward_settings():
    name = 'stage2_sensor_900_bridge_before_fixed_layout_cache'
    cfg = native_config(name)
    train = native_config(name, True)
    validate_experiment(name, cfg['env'], train)
    base = native_config('stage2_sensor_900_bridge_before_fixed')
    env = deepcopy(cfg['env'])
    bridge = env['sensorScene']['bridge']
    assert bridge.pop('layout_cache') == dict(entries_per_bucket=2, refresh_probability=0.)
    assert bridge['distance_curriculum']['mixture'] == [0., 0., 1.]
    bridge['distance_curriculum'] = deepcopy(base['env']['sensorScene']['bridge']['distance_curriculum'])
    assert env == base['env']
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    wrong = deepcopy(cfg['env'])
    wrong['sensorScene']['bridge']['layout_cache']['entries_per_bucket'] = 0
    from utils.sensor_stage2_spec import validate_sensor
    with pytest.raises(ValueError):
        validate_sensor(wrong)
    wrong = deepcopy(base['env'])
    wrong['sensorScene']['bridge']['distance_curriculum']['mixture'] = [0., 0., 1.]
    with pytest.raises(ValueError):
        validate_sensor(wrong)
