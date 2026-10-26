from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from utils.before_sampling import (DISTANCE_MIXTURE, THREE_WAY_DISTANCE_MIXTURE,
    sample_near_pairs, sample_follower_xy, sample_distance_modes,
    sample_follower_modes, shorten_leader_xy)
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage1_sampling import sample_source_xy
from utils.stage2_spec import check_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def test_mix_config_changes_only_sampling_and_isolates_checkpoint():
    directory = ROOT / 'tokenhsi/data/cfg/multi_agent'
    base = yaml.safe_load((directory / 'stage2_before_ground.yaml').read_text())['env']
    mixed = yaml.safe_load((directory / 'stage2_before_ground_mix50.yaml').read_text())['env']
    validate_env(base)
    validate_env(mixed)
    comparable = deepcopy(mixed)
    assert comparable['before'].pop('distance_mixture') == DISTANCE_MIXTURE
    assert comparable == base
    check_checkpoint({'before_contract': checkpoint_contract(mixed)}, mixed)
    for source, target in ((base, mixed), (mixed, base)):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    bad = deepcopy(mixed)
    bad['before']['distance_mixture']['near_probability'] = .7
    with pytest.raises(ValueError, match='50/50'):
        validate_env(bad)
    directory = ROOT / 'tokenhsi/data/cfg/train/rlg'
    old = yaml.safe_load((directory / 'stage2_before_ground.yaml').read_text())
    new = yaml.safe_load((directory / 'stage2_before_ground_mix50.yaml').read_text())
    new['params']['config']['name'] = old['params']['config']['name']
    assert new == old


def test_modes_are_per_pair_half_and_solo_is_untouched():
    pre = torch.tensor([-1, 0, -1, 2, -1]).expand(12000, -1)
    torch.manual_seed(51)
    near = sample_near_pairs(pre, DISTANCE_MIXTURE)
    assert not near[pre < 0].any()
    assert .48 < near[pre >= 0].float().mean() < .52
    assert (near[:, 1] != near[:, 3]).any()  # Independent pairs in one scene.
    torch.manual_seed(51)
    assert torch.equal(near, sample_near_pairs(pre, DISTANCE_MIXTURE))
    rng = torch.get_rng_state()
    assert not sample_near_pairs(pre, None).any()
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('template', [1, 2, 4])
def test_mixed_distances_and_ontop_payload_reference(template):
    n = 10000
    center = torch.tensor([12., -8.]).expand(n, -1)
    templates = torch.full((n,), template)
    near = torch.arange(n) % 2 == 0
    torch.manual_seed(5100)
    follower = sample_follower_xy(center, templates, near)
    distance = (follower-center).norm(dim=-1)
    assert (distance >= 1.-1e-5).all()
    assert (distance[near] <= 5.+1e-5).all()
    maximum = 5. if template == 1 else 10.
    assert (distance[~near] <= maximum+1e-5).all()
    assert distance[~near].max() > maximum-.1
    # The close range retains a broad radius distribution, not just its endpoints.
    assert 2.9 < distance[near].mean() < 3.1
    if template == 4:
        payload = sample_follower_xy(follower, templates, near)
        payload_distance = (payload-follower).norm(dim=-1)
        assert (payload_distance >= 1.-1e-5).all()
        assert (payload_distance[near] <= 5.+1e-5).all()
        assert payload_distance[~near].max() > 9.9
        assert (payload[near]-center[near]).norm(dim=-1).max() > 5.
    # No ordering constraint against the independently placed leader.
    leader_distance = 1.+9.*torch.rand(n)
    assert (distance[near] > leader_distance[near]).any()
    assert (distance[near] < leader_distance[near]).any()


def test_disabled_sampling_preserves_values_and_rng():
    center = torch.zeros(100, 2)
    templates = torch.arange(100) % 4 + 1
    torch.manual_seed(8)
    expected = sample_source_xy(center, templates)
    rng = torch.get_rng_state()
    torch.manual_seed(8)
    actual = sample_follower_xy(center, templates, torch.zeros(100, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_force_near_selects_all_before_pairs_without_mutating_config():
    pre = torch.tensor([[-1, 0, -1, 2, -1], [-1, -1, -1, -1, -1]])
    mixture = deepcopy(DISTANCE_MIXTURE)
    state = torch.get_rng_state()
    near = sample_near_pairs(pre, mixture, force_near=True)
    assert torch.equal(near, pre >= 0)
    assert torch.equal(state, torch.get_rng_state())
    assert mixture == DISTANCE_MIXTURE
    # Existing non-mixture viewers retain their separate layout implementation.
    assert not sample_near_pairs(pre, None, force_near=True).any()


def test_three_way_config_and_checkpoint_isolation():
    directory = ROOT / 'tokenhsi/data/cfg/multi_agent'
    old = yaml.safe_load((directory/'stage2_before_ground_mix50.yaml').read_text())['env']
    new = yaml.safe_load((directory/'stage2_before_ground_mix334_1200.yaml').read_text())['env']
    validate_env(new)
    assert new['episodeLength'] == 1200
    comparable = deepcopy(new)
    assert comparable['before'].pop('distance_mixture') == THREE_WAY_DISTANCE_MIXTURE
    comparable['before']['distance_mixture'] = DISTANCE_MIXTURE
    comparable['episodeLength'] = 600
    assert comparable == old
    for source, target in ((old, new), (new, old)):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    bad = deepcopy(new)
    bad['before']['distance_mixture']['probabilities'] = [.2, .4, .4]
    with pytest.raises(ValueError, match='30/30/40'):
        validate_env(bad)


def test_three_way_mode_frequencies_solo_and_legacy_rng():
    pre = torch.tensor([-1, 0, -1, 2, -1]).expand(20000, -1)
    torch.manual_seed(334)
    modes = sample_distance_modes(pre, THREE_WAY_DISTANCE_MIXTURE)
    assert not modes[pre < 0].any()
    frequencies = torch.bincount(modes[pre >= 0], minlength=3) / 40000.
    torch.testing.assert_close(frequencies, torch.tensor([.3, .3, .4]), atol=.015, rtol=0)
    assert (modes[:, 1] != modes[:, 3]).any()
    torch.manual_seed(334)
    assert torch.equal(modes, sample_distance_modes(pre, THREE_WAY_DISTANCE_MIXTURE))
    assert torch.equal(sample_distance_modes(pre, THREE_WAY_DISTANCE_MIXTURE, True), (pre >= 0).long())
    torch.manual_seed(12)
    old = sample_near_pairs(pre, DISTANCE_MIXTURE)
    state = torch.get_rng_state()
    torch.manual_seed(12)
    assert torch.equal(sample_distance_modes(pre, DISTANCE_MIXTURE), old.long())
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('template', [1, 2, 4])
def test_three_way_distances(template):
    n = 15000
    centers = torch.tensor([7., -13.]).expand(n, -1)
    modes = torch.arange(n) % 3
    templates = torch.full((n,), template)
    xy = sample_follower_modes(centers, templates, modes)
    dist = (xy-centers).norm(dim=-1)
    for mode, maximum in enumerate((5. if template == 1 else 10., 5., 2.5)):
        selected = dist[modes == mode]
        assert selected.min() >= 1.-1e-5
        assert selected.max() <= maximum+1e-5
        assert selected.max() > maximum-.05
        assert abs(selected.mean()-(1.+maximum)/2.) < .1
    payload = sample_follower_modes(xy, torch.full_like(templates, 4), modes)
    own = (payload-xy).norm(dim=-1)
    assert own[modes == 2].max() <= 2.5+1e-5
    assert own[modes == 0].max() > 9.9
    # Old mixture values and RNG remain identical when short mode is absent.
    modes = modes.clamp_max(1)
    torch.manual_seed(33)
    expected = sample_follower_xy(centers, templates, modes.bool())
    state = torch.get_rng_state()
    torch.manual_seed(33)
    torch.testing.assert_close(sample_follower_modes(centers, templates, modes), expected, atol=0, rtol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_short_leader_keeps_payload_fixed_and_preserves_direction():
    box = torch.tensor([[3., -2.]]).expand(1000, -1).clone()
    original = box.clone()
    radius = torch.linspace(1., 10., 1000)
    human = box + radius[:, None] * torch.tensor([[.6, .8]])
    state = torch.get_rng_state()
    moved = shorten_leader_xy(human, box)
    torch.testing.assert_close((moved-box).norm(dim=-1), torch.linspace(1., 5., 1000))
    torch.testing.assert_close((moved-box)/(moved-box).norm(dim=-1, keepdim=True), torch.tensor([[.6, .8]]).expand(1000, -1))
    assert torch.equal(box, original) and torch.equal(state, torch.get_rng_state())


def test_force_short_selects_only_before_pairs_without_rng_or_config_changes():
    pre = torch.tensor([[-1, 0, -1, 2], [-1, -1, -1, -1]])
    mixture = deepcopy(THREE_WAY_DISTANCE_MIXTURE)
    state = torch.get_rng_state()
    modes = sample_distance_modes(pre, mixture, force_short=True)
    assert torch.equal(modes, torch.tensor([[0, 2, 0, 2], [0, 0, 0, 0]]))
    assert torch.equal(state, torch.get_rng_state())
    assert mixture == THREE_WAY_DISTANCE_MIXTURE
    for bad in (None, DISTANCE_MIXTURE):
        with pytest.raises(ValueError, match='30/30/40'):
            sample_distance_modes(pre, bad, force_short=True)
    with pytest.raises(ValueError, match='force_near'):
        sample_distance_modes(pre, mixture, force_near=True, force_short=True)
