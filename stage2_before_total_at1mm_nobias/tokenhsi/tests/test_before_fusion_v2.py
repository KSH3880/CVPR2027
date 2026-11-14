from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.before_wait import BeforeWait
from test_before_wait import setup
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg'
NAME = BASE + '_v2'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_penalty']


def test_config_only_changes_waiting_entry_and_experiment_name():
    cfg, base = native_config(NAME), native_config(BASE)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert OPTION == {'radius_buffer': 2.5, 'coefficient': 1., 'ontop_distance_source': 'human'}
    cfg['env']['before']['waiting_penalty'] = deepcopy(base['env']['before']['waiting_penalty'])
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    check_checkpoint({'before_contract': checkpoint_contract(ENV)}, ENV)
    for source, target in ((ENV, base['env']), (base['env'], ENV)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_root_boundary_xyz_penalty_latch_release_and_partial_reset(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    center = goals[rows, state.goal, :2].clone()
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2 + 2.5
    roots[rows, state.owners, :2] = center
    roots[rows, state.owners, 0] += radius + .01
    assert not state.penalty(*args).any()
    roots[rows, state.owners, 0] -= .02
    goals[..., 2] += 100
    velocity.zero_(); velocity[..., 2] = .5
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .5
    snapshots = [v.clone() for v in args]
    torch.testing.assert_close(state.penalty(*args), expected)
    assert all(torch.equal(a, b) for a, b in zip(args, snapshots))
    roots[rows, state.owners, :2] += 100
    velocity[:] = torch.tensor([.3, .4, 0.])
    torch.testing.assert_close(state.penalty(*args), expected)
    raw[rows[rows == 0], state.predecessor[rows == 0]] = True
    expected[0] = 0
    torch.testing.assert_close(state.penalty(*args), expected)
    raw.zero_()
    torch.testing.assert_close(state.penalty(*args), expected)
    state.reset(torch.tensor([1]), graph, raw)
    expected[1] = 0
    torch.testing.assert_close(state.penalty(*args), expected)
    assert not state.age.any() and not state.moving_steps.any()


@pytest.mark.parametrize('root_near', [False, True])
def test_ontop_entry_uses_person_even_when_payload_is_elsewhere(root_near):
    state, args, graph, rows, edges = setup('stack', config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    center = goals[rows, state.goal, :2]
    roots[rows, state.owners, :2] = center + (0 if root_near else 20)
    boxes[rows, state.source-4, :2] = center + (20 if root_near else 0)
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = float(root_near)
    torch.testing.assert_close(state.penalty(*args), expected)
    legacy = BeforeWait(graph, len(roots), .1, native_config(BASE)['env']['before']['waiting_penalty'], 'cpu')
    legacy.reset(torch.arange(len(roots)), graph, raw)
    expected[rows, state.owners] = float(not root_near)
    torch.testing.assert_close(legacy.penalty(*args), expected)


@pytest.mark.parametrize('value', ['root', None, True, 1])
def test_invalid_ontop_distance_source(value):
    env = deepcopy(ENV)
    env['before']['waiting_penalty']['ontop_distance_source'] = value
    with pytest.raises(ValueError, match='waiting penalty'):
        validate_unified_env(env)
