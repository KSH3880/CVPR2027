from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config, validate_experiment, check_before_checkpoint
from utils.before_push_spec import sample_graph
from utils.before_spec import checkpoint_contract
from utils.before_sampling import sample_near_pairs
from utils.before_relative_sampling import sample_layout, pair_distances
from utils.stage1_sampling import sample_source_xy
from utils.unified_training import validate_unified_env

NAME = 'stage2_before_push_near50_pair50_source5'
BASE = 'stage2_before_push_near50'


def test_only_pair_coin_and_source_distances_change():
    base = native_config(BASE)
    cfg = native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg)
    restored['experiment']['name'] = BASE
    restored['env']['before']['pairing_probability'] = .8
    restored['env']['before'].pop('source_max_distance')
    restored['env']['jointPush']['scene']['source_distance'] = [2., 10.]
    assert restored == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    assert cfg['env']['jointPush']['scene']['goal_distance'] == [2., 10.]


def test_pair50_distribution_preserves_solo_prior_and_near50():
    torch.set_num_threads(1)
    torch.manual_seed(53)
    env = native_config(NAME)['env']
    _, pre, templates = sample_graph(torch.full((40000, 4, 3), .4),
        env['box']['build']['taskSizeRanges'], env['relationGraph']['template_probabilities'],
        pairing_probability=env['before']['pairing_probability'])
    counts = torch.bincount((pre >= 0).sum(-1), minlength=3)/len(pre)
    torch.testing.assert_close(counts, torch.tensor([.125, .625, .25]), atol=.01, rtol=0)
    paired = pre >= 0
    r, f = paired.nonzero(as_tuple=True)
    paired[r, pre[r, f]] = True
    solo = torch.bincount(templates[~paired], minlength=6)[1:]/(~paired).sum()
    torch.testing.assert_close(solo, torch.tensor([.1, .15, .25, .2, .3]), atol=.01, rtol=0)
    near = sample_near_pairs(pre, env['before']['distance_mixture'])
    assert .49 < near[pre >= 0].float().mean() < .51


def test_limited_pair_geometry_and_unrestricted_ontop_transport():
    torch.manual_seed(54)
    templates = torch.tensor([1, 2, 4]).repeat(2000)
    roots = torch.zeros(len(templates), 2)
    support = sample_source_xy(roots, torch.full_like(templates, 3), 5.)
    near = torch.arange(len(templates)) % 2 == 0
    box, follower, payload = sample_layout(roots, support, roots, roots, templates,
        near, torch.zeros_like(near), max_distance=5.)
    first, second = pair_distances(roots, box, follower, payload, templates)
    for distance in (first, (follower-box).norm(dim=-1), (follower-payload).norm(dim=-1)):
        assert ((distance >= 1.-2e-5) & (distance <= 5.+2e-5)).all()
    assert (second[near] < first[near]).all()
    top_original = (templates == 4) & ~near
    assert ((payload-box).norm(dim=-1)[top_original] > 5.).any()


def test_default_pair_sampler_retains_exact_rng_and_results():
    env = native_config(BASE)['env']
    args = (torch.full((256, 4, 3), .4), env['box']['build']['taskSizeRanges'],
            env['relationGraph']['template_probabilities'])
    torch.manual_seed(55)
    _, pre, templates = sample_graph(*args)
    after = torch.get_rng_state()
    torch.manual_seed(55)
    _, explicit_pre, explicit_templates = sample_graph(*args, pairing_probability=.8)
    assert torch.equal(pre, explicit_pre) and torch.equal(templates, explicit_templates)
    assert torch.equal(after, torch.get_rng_state())


def test_push_retry_preserves_accepted_layouts_and_identifies_blocked_scene():
    from utils.before_push_layout import sample_layout, paths_valid, PushLayoutError
    torch.manual_seed(56)
    boxes = torch.zeros(2, 2, 13)
    boxes[..., 6] = 1.
    boxes[:, 0, :2] = 1000.
    sizes = torch.full((2, 2, 3), .4)
    sizes[1, 1] = 100.
    roots = torch.zeros(2, 1, 3)
    push = torch.ones(2, 1, dtype=torch.bool)
    primary = torch.zeros(2, 1, dtype=torch.long)
    active = torch.tensor([[True, False], [True, True]])
    spec = native_config(NAME)['env']['jointPush']['scene']
    with pytest.raises(PushLayoutError) as caught:
        sample_layout(boxes, sizes, roots, push, primary, active, spec, max_attempts=4)
    failed = caught.value
    assert failed.pending.tolist() == [1]
    assert not torch.equal(failed.boxes[0], boxes[0])
    assert paths_valid(failed.boxes[:1], sizes[:1], failed.goals[:1], roots[:1],
        push[:1], primary[:1], active[:1], spec).all()


@pytest.mark.parametrize('other', [BASE, 'stage2_before_push'])
def test_checkpoint_contracts_cannot_be_mixed(other):
    new, old = native_config(NAME)['env'], native_config(other)['env']
    for source, target in ((new, old), (old, new)):
        with pytest.raises(ValueError):
            check_before_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    with pytest.raises(ValueError):
        validate_experiment(NAME, old, native_config(NAME, True))


@pytest.mark.parametrize('key,value', [('pairing_probability', .6), ('source_max_distance', 6.)])
def test_rejects_unregistered_sampling_drift(key, value):
    env = native_config(NAME)['env']
    env['before'][key] = value
    with pytest.raises(ValueError):
        validate_unified_env(env)
