from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from utils.before_sampling import (MIX226_DISTANCE_MIXTURE, THREE_WAY_DISTANCE_MIXTURE,
    sample_distance_modes, sample_follower_modes, sample_before_goal_xy)
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def test_config_changes_only_distance_mixture_and_isolates_checkpoints():
    directory = ROOT/'tokenhsi/data/cfg/multi_agent'
    old = yaml.safe_load((directory/'stage2_before_ground_mix334_1200.yaml').read_text())['env']
    new = yaml.safe_load((directory/'stage2_before_ground_mix226_1200.yaml').read_text())['env']
    validate_env(new)
    comparable = deepcopy(new)
    assert comparable['before'].pop('distance_mixture') == MIX226_DISTANCE_MIXTURE
    comparable['before']['distance_mixture'] = THREE_WAY_DISTANCE_MIXTURE
    assert comparable == old
    check_checkpoint({'before_contract': checkpoint_contract(new)}, new)
    for source, target in ((old, new), (new, old)):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    for key in ('payload_distance', 'ontop_follower_distance', 'leader_goal_distance'):
        bad = deepcopy(new)
        bad['before']['distance_mixture'][key][2][1] += .1
        with pytest.raises(ValueError, match='20/20/60'):
            validate_env(bad)
    directory = ROOT/'tokenhsi/data/cfg/train/rlg'
    old = yaml.safe_load((directory/'stage2_before_ground_mix334_1200.yaml').read_text())
    new = yaml.safe_load((directory/'stage2_before_ground_mix226_1200.yaml').read_text())
    new['params']['config']['name'] = old['params']['config']['name']
    assert new == old


def test_pair_frequencies_independence_solo_and_viewer_overrides():
    torch.manual_seed(226)
    pre = torch.tensor([-1, 0, -1, 2, -1]).expand(20000, -1)
    modes = sample_distance_modes(pre, MIX226_DISTANCE_MIXTURE)
    assert not modes[pre < 0].any()
    torch.testing.assert_close(torch.bincount(modes[pre >= 0], minlength=3)/40000.,
                               torch.tensor([.2, .2, .6]), atol=.012, rtol=0)
    assert (modes[:, 1] != modes[:, 3]).any()
    state = torch.get_rng_state()
    assert torch.equal(sample_distance_modes(pre, MIX226_DISTANCE_MIXTURE, force_near=True), (pre >= 0).long())
    assert torch.equal(sample_distance_modes(pre, MIX226_DISTANCE_MIXTURE, force_short=True), (pre >= 0).long()*2)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('template,maxima', [(1, [5.,5.,2.5]), (2, [10.,5.,2.5]), (4, [5.,2.5,1.25])])
def test_all_follower_ranges_and_ontop_payload_center(template, maxima):
    torch.manual_seed(226+template)
    centers = torch.tensor([9., -11.]).expand(12000, -1)
    modes = torch.arange(len(centers)) % 3
    templates = torch.full_like(modes, template)
    follower = sample_follower_modes(centers, templates, modes, MIX226_DISTANCE_MIXTURE)
    groups = [(follower-centers).norm(dim=-1)]
    if template == 4:
        payload = sample_follower_modes(follower, templates, modes, MIX226_DISTANCE_MIXTURE, payload=True)
        groups.append((payload-follower).norm(dim=-1))
        assert (payload-centers).norm(dim=-1)[modes == 2].max() > 2.
    for distance in groups:
        for mode, maximum in enumerate(maxima):
            selected = distance[modes == mode]
            assert selected.min() >= 1.-1e-5
            assert maximum-.02 < selected.max() <= maximum+1e-5
            assert abs(selected.mean()-(1.+maximum)/2.) < .08


def test_transport_is_box_centered_with_human_exclusion_and_translation_invariance():
    box = torch.tensor([14., -9.]).expand(15000, -1)
    human = box + torch.tensor([1.5, 0.])
    modes = torch.arange(len(box)) % 3
    torch.manual_seed(226)
    goal = sample_before_goal_xy(human, box, modes, MIX226_DISTANCE_MIXTURE)
    distance = (goal-box).norm(dim=-1)
    assert ((goal-human).norm(dim=-1) >= 1.-1e-5).all()
    for mode, maximum in enumerate((10., 10., 5.)):
        selected = distance[modes == mode]
        assert selected.min() >= 1.-1e-5
        assert maximum-.03 < selected.max() <= maximum+1e-5
    assert ((goal-box).abs() > 4.5).any()  # No environment-square clipping.
    shift = torch.tensor([32., 16.])
    torch.manual_seed(226)
    moved = sample_before_goal_xy(human+shift, box+shift, modes, MIX226_DISTANCE_MIXTURE)
    torch.testing.assert_close(moved, goal+shift, atol=1e-5, rtol=0)


def test_old_mixture_call_path_preserves_values_and_rng():
    center = torch.randn(90, 2)
    templates = torch.arange(90) % 4 + 1
    modes = torch.arange(90) % 3
    torch.manual_seed(334)
    expected = sample_follower_modes(center, templates, modes)
    state = torch.get_rng_state()
    torch.manual_seed(334)
    actual = sample_follower_modes(center, templates, modes, THREE_WAY_DISTANCE_MIXTURE)
    assert torch.equal(expected, actual) and torch.equal(state, torch.get_rng_state())
