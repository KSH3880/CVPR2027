from copy import deepcopy

import pytest
import torch

from test_before_wait import BASE, setup
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract, check_checkpoint

NAME = BASE + '_wait_penalty'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_penalty']


def test_config_preserves_virtual_except_penalty_and_at_height():
    cfg, base = native_config(NAME), native_config(BASE)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg)
    assert restored['env']['before'].pop('waiting_penalty') == {'radius_buffer': 2., 'coefficient': .6}
    restored['env']['relationReward']['success']['at_z_tolerance'] = .001
    restored['experiment']['name'] = BASE
    assert restored == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    for other in (BASE, BASE+'_wait_v3'):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(native_config(other)['env'])}, ENV)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_continuous_xyz_penalty_latched_entry_release_and_partial_reset(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    entity = boxes if family == 'stack' else roots
    index = state.source-4 if family == 'stack' else state.owners
    center = goals[rows, state.goal, :2].clone()
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2 + 2.
    entity[rows, index, :2] = center
    entity[rows, index, 0] += radius + .01
    assert not state.penalty(*args).any()
    entity[rows, index, 0] -= .02
    goals[..., 2] += 100
    for speed in (0., .1, .3, .5, 1., 2.):
        velocity.zero_(); velocity[..., 2] = speed
        expected = torch.zeros(roots.shape[:2])
        expected[rows, state.owners] = .6*speed
        snapshots = [v.clone() for v in args]
        torch.testing.assert_close(state.penalty(*args), expected)
        assert all(torch.equal(a, b) for a, b in zip(args, snapshots))
    entity[rows, index, 0] += 100
    velocity[:] = torch.tensor([.3, .4, 0.])
    expected[rows, state.owners] = .3
    torch.testing.assert_close(state.penalty(*args), expected)
    # Release only one pair, and keep it released when current success disappears.
    chosen = rows == 0
    raw[rows[chosen], state.predecessor[chosen]] = True
    expected[rows[chosen], state.owners[chosen]] = 0
    torch.testing.assert_close(state.penalty(*args), expected)
    raw.zero_()
    torch.testing.assert_close(state.penalty(*args), expected)
    state.reset(torch.tensor([1]), graph, raw)
    expected[1] = 0
    torch.testing.assert_close(state.penalty(*args), expected)
    assert not state.age.any() and not state.moving_steps.any()


@pytest.mark.parametrize('key,value', [('coefficient', 0), ('coefficient', float('nan')),
    ('radius_buffer', -1), ('coefficient', True), ('extra', 1)])
def test_invalid_penalty(key, value):
    env = deepcopy(ENV)
    env['before']['waiting_penalty'][key] = value
    with pytest.raises(ValueError, match='waiting penalty'):
        validate_unified_env(env)


def test_penalty_and_termination_cannot_be_enabled_together():
    env = deepcopy(ENV)
    env['before']['waiting_termination'] = native_config(BASE+'_wait_v3')['env']['before']['waiting_termination']
    with pytest.raises(ValueError, match='waiting penalty'):
        validate_unified_env(env)
