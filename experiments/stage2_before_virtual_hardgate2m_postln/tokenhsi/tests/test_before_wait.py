from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.before_wait import BeforeWait
from utils.before_push_spec import sample_graph
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.edge_ontop_spec import permute_graph
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from test_before_atprogress import fixture

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute'
NAME = BASE + '_wait'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_termination']


def setup(family, dt=.1):
    graph, roots, boxes, sizes, goals, rows, edges, goal_ids, support = fixture(family)
    raw = torch.zeros_like(graph.edge_valid)
    state = BeforeWait(graph, len(roots), dt, OPTION, 'cpu')
    state.reset(torch.arange(len(roots)), graph, raw)
    xy = goals[rows, goal_ids, :2]
    if family == 'stack':
        boxes[rows, graph.edge_src[rows, edges]-4, :2] = xy
    else:
        roots[rows, graph.edge_owner[rows, edges], :2] = xy
    velocity = torch.zeros_like(roots)
    velocity[..., 0] = 1.
    return state, (roots, velocity, boxes, sizes, goals, raw), graph, rows, edges


def test_config_preserves_base_and_separates_contract():
    base, cfg = native_config(BASE), native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg['env'])
    assert restored['before'].pop('waiting_termination') == OPTION
    assert restored == base['env'] and cfg['sim'] == base['sim']
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    train = native_config(NAME, True)
    assert train['params']['config']['name'] == native_config(BASE, True)['params']['config']['name']+'Wait'
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    for source, target in ((base, cfg), (cfg, base)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])


@pytest.mark.parametrize('key', OPTION)
@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan'), True, '1', None])
def test_invalid_waiting_parameters(key, value):
    env = deepcopy(ENV)
    env['before']['waiting_termination'][key] = value
    with pytest.raises(ValueError, match='waiting termination'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
@pytest.mark.parametrize('dt', [.1, 1/30])
def test_grace_then_continuous_movement_only_terminates_follower(family, dt):
    state, args, graph, rows, edges = setup(family, dt)
    owners = graph.edge_owner[rows, edges]
    snapshots = [value.clone() for value in args]
    for _ in range(state.grace_steps+state.violation_steps-1):
        assert not state.step(*args).any()
    result = state.step(*args)
    expected = torch.zeros_like(result)
    expected[rows, owners] = True
    assert torch.equal(result, expected)
    assert all(torch.equal(a, b) for a, b in zip(args, snapshots))


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_stop_resets_counter_but_leaving_radius_does_not_disarm(family):
    state, args, graph, rows, edges = setup(family)
    for _ in range(state.grace_steps+state.violation_steps-1):
        assert not state.step(*args).any()
    args[1].zero_()
    assert not state.step(*args).any()
    assert not state.moving_steps.any()
    args[0][..., :2] += 100
    args[2][..., :2] += 100
    args[1][..., 0] = 1.
    for _ in range(state.violation_steps-1):
        assert not state.step(*args).any()
    assert state.step(*args).sum() == len(rows)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_same_frame_success_releases_permanently_and_partial_reset_is_local(family):
    state, args, graph, rows, edges = setup(family)
    for _ in range(state.grace_steps+state.violation_steps-1):
        state.step(*args)
    leader = graph.prereq_mask[rows, edges].long().argmax(-1)
    args[-1][rows, leader] = True
    assert not state.step(*args).any()
    args[-1].zero_()
    for _ in range(30):
        assert not state.step(*args).any()
    assert state.released[rows, edges].all()
    state.reset(torch.tensor([1]), graph, args[-1])
    assert not state.armed[1].any() and not state.released[1].any()
    assert state.released[rows[rows != 1], edges[rows != 1]].all()
    for _ in range(state.grace_steps+state.violation_steps-1):
        assert not state.step(*args).any()
    result = state.step(*args)
    assert result.sum() == 1 and result[1].any()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_xy_boundary_and_source_entity(family):
    state, args, graph, rows, edges = setup(family)
    roots, velocity, boxes, sizes, goals, raw = args
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2 + .5
    xy = torch.stack((radius+.01, torch.zeros_like(radius)), -1)
    owners = graph.edge_owner[rows, edges]
    if family == 'stack':
        boxes[rows, state.source-4, :2] = xy
    else:
        roots[rows, owners, :2] = xy
    assert not state.step(*args).any() and not state.armed.any()
    goals[..., 2] += 100
    if family == 'stack':
        roots[..., :2] += 50
        boxes[rows, state.source-4, 0] -= .01
    else:
        boxes[..., :2] += 50
        roots[rows, owners, 0] -= .01
    state.step(*args)
    assert state.armed[rows, edges].all()
    velocity.zero_()
    velocity[..., 0] = .3
    for _ in range(30):
        assert not state.step(*args).any()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_multiple_pairs_shuffled_edges_bind_their_own_goal_and_success(family):
    graph, _, _ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
        ENV['relationGraph']['template_probabilities'], 'before_'+family)
    graph = permute_graph(graph, torch.rand(3, 8).argsort(-1))
    raw = torch.zeros_like(graph.edge_valid)
    state = BeforeWait(graph, 3, .1, OPTION, 'cpu')
    state.reset(torch.arange(3), graph, raw)
    rows, edges = state.rows, state.edges
    roots, boxes, sizes = torch.zeros(3, 4, 3), torch.zeros(3, 8, 13), torch.full((3, 8, 3), .4)
    goals = torch.zeros(3, 4, 3)
    goals[..., 0] = torch.arange(4)*4.
    if family == 'stack':
        boxes[rows, state.source-4, :2] = goals[rows, state.goal, :2]
    else:
        roots[rows, state.owners, :2] = goals[rows, state.goal, :2]
    raw[rows[::2], state.predecessor[::2]] = True
    velocity = torch.ones_like(roots)
    for _ in range(20):
        result = state.step(roots, velocity, boxes, sizes, goals, raw)
    expected = torch.zeros_like(result)
    expected[rows[1::2], state.owners[1::2]] = True
    assert len(rows) == 6 and torch.equal(result, expected)


@pytest.mark.parametrize('preset', ['all_solo', 'push'])
def test_independent_tasks_never_arm(preset):
    graph, _, _ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
        ENV['relationGraph']['template_probabilities'], preset)
    state = BeforeWait(graph, 3, .1, OPTION, 'cpu')
    raw = torch.zeros_like(graph.edge_valid)
    state.reset(torch.arange(3), graph, raw)
    for _ in range(25):
        assert not state.step(torch.zeros(3, 4, 3), torch.ones(3, 4, 3), torch.zeros(3, 8, 13),
            torch.ones(3, 8, 3), torch.zeros(3, 4, 3), raw).any()
    assert not state.armed.any()
