from copy import deepcopy
import math

import pytest
import torch

from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from utils.before_push_spec import sample_graph
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.edge_ontop_spec import permute_graph, select_graph
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from test_before_atprogress import fixture

BASE = 'stage2_before_push_near50_pair50_source5_virtual'
NAME = BASE + '_asym'
ENV = native_config(NAME)['env']
CONFIG = ENV['relationReward']


def geometry(family, offsets):
    g, roots, boxes, sizes, goals, rows, edges, goal_ids, support = fixture(family)
    radius = sizes[rows, support, :2].norm(dim=-1)/2 + (.1 if family == 'stack' else .3)
    distance = radius + torch.tensor(offsets)
    source = g.edge_src[rows, edges]
    positions = boxes[rows, source-4, :2] if family == 'stack' else roots[rows, source, :2]
    positions[:, 0] = distance
    positions[:, 1] = 0
    if family == 'stack':
        boxes[rows, source-4, :2] = positions
    else:
        roots[rows, source, :2] = positions
    return (roots, boxes, sizes, goals, g, CONFIG), rows, edges, radius


def test_config_only_adds_asymmetric_progress_and_separates_checkpoints():
    base, cfg = native_config(BASE), native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg['env'])
    assert restored['before'].pop('asymmetric_progress') == {'lambda_inner': 10.}
    assert restored == base['env'] and cfg['sim'] == base['sim']
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    for source, target in ((base, cfg), (cfg, base)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])
        with pytest.raises(ValueError, match='selected experiment'):
            validate_experiment(source['experiment']['name'], target['env'], native_config(source['experiment']['name'], True))


@pytest.mark.parametrize('value', [-1., float('inf'), float('nan'), True, '10', None])
def test_invalid_lambda_is_rejected(value):
    env = deepcopy(ENV)
    env['before']['asymmetric_progress']['lambda_inner'] = value
    with pytest.raises(ValueError, match='Asymmetric progress'):
        validate_unified_env(env)


@pytest.mark.parametrize('option', [None, {}, {'lambda_inner': 10., 'extra': True}])
def test_invalid_asymmetric_option_is_rejected(option):
    env = deepcopy(ENV)
    env['before']['asymmetric_progress'] = option
    with pytest.raises(ValueError, match='Asymmetric progress'):
        validate_unified_env(env)


def test_asymmetric_requires_virtual_and_accepts_zero_or_five_for_comparison():
    for value in (0., 5.):
        env = deepcopy(ENV)
        env['before']['asymmetric_progress']['lambda_inner'] = value
        validate_unified_env(env)
    env['relationGraph'].pop('before_virtual_object')
    with pytest.raises(ValueError, match='Asymmetric progress'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_outer_boundary_inner_and_success_recovery(family):
    args, rows, edges, radius = geometry(family, [1., 0., -.05, -.2])
    g = args[-2]
    original = torch.rand(4, len(g.ids))
    saved = original.clone()
    old = predecessor_at_progress(original, *args)
    failed = torch.zeros_like(original, dtype=torch.bool)
    new = predecessor_at_progress(original, *args, raw_success=failed, lambda_inner=10.)
    assert new[rows[0], edges[0]] == old[rows[0], edges[0]]
    assert new[rows[1], edges[1]] == 1
    torch.testing.assert_close(new[rows[2:], edges[2:]], torch.tensor([1/1.5, 1/3.]))
    assert new[rows[3], edges[3]] < new[rows[2], edges[2]] < 1
    assert torch.equal(new[~g.prereq_mask.any(-1)], original[~g.prereq_mask.any(-1)])
    assert torch.equal(original, saved)
    assert torch.equal(predecessor_at_progress(original, *args, raw_success=~failed, lambda_inner=10.), old)
    assert torch.equal(predecessor_at_progress(original, *args, lambda_inner=0.), old)
    assert torch.equal(predecessor_at_progress(original, *args, raw_success=failed, lambda_inner=10.), new)
    args[3][..., 2] += 100
    assert torch.equal(predecessor_at_progress(original, *args, raw_success=failed, lambda_inner=10.), new)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_current_at_predicate_gate_holding_and_success_saturation(family):
    args, rows, edges, _ = geometry(family, [-.1]*4)
    graph = args[-2]
    old = predecessor_at_progress(torch.full((4, len(graph.ids)), .4), *args)
    runtime = BeforePushRuntime(4, graph, CONFIG, 'cpu', contact_weights=ENV['jointPush']['contact_weights'])
    leader = graph.prereq_mask[rows, edges].long().argmax(-1)
    runtime.achieved.fill_(True)
    runtime.own_success.fill_(True)
    for at_distance, at_z, expected_success in ((.3, 0., False), (.1, 0., True), (.1, .002, False), (.1, 0., True)):
        phi = torch.full_like(old, .1)
        phi[rows, leader] = math.exp(-10*at_distance**2)
        z = torch.ones_like(phi)
        z[rows, leader] = at_z
        errors = torch.ones_like(phi)
        raw = interaction_own_success(phi, z, errors, graph, CONFIG, errors)
        progress = predecessor_at_progress(old, *args, raw_success=raw, lambda_inner=10.)
        actual = runtime.components(phi, progress, z, errors, errors, graph)
        previous = runtime.components(phi, old, z, errors, errors, graph)
        assert (raw[rows, leader] == expected_success).all()
        assert torch.equal(raw, actual['raw_own_success'])
        assert (actual['prerequisite_gate'][rows, edges] == expected_success).all()
        expected = torch.ones(4) if expected_success else torch.full((4,), .5)
        torch.testing.assert_close(progress[rows, edges], expected)
        torch.testing.assert_close(actual['progress_component'][rows, edges], .2*expected)
        for key in ('state_component', 'success_component', 'own_success', 'reward_saturated'):
            assert torch.equal(actual[key], previous[key])
        other = ~graph.prereq_mask.any(-1)
        assert torch.equal(actual['total'][other], previous['total'][other])
    ones, zeros = torch.ones_like(old), torch.zeros_like(old)
    progress = predecessor_at_progress(old, *args, raw_success=ones.bool(), lambda_inner=10.)
    saturated = runtime.components(ones, progress, zeros, zeros, zeros, graph)
    assert (saturated['progress_component'][rows, edges] == .2).all()
    assert (saturated['success_component'][rows, edges] == .2).all()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_multiple_pairs_permutation_and_partial_batch_use_each_predecessor(family):
    g, _, _ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
        ENV['relationGraph']['template_probabilities'], 'before_'+family)
    rows, edges = g.prereq_mask.any(-1).nonzero(as_tuple=True)
    leader = g.prereq_mask[rows, edges].long().argmax(-1)
    goal = g.edge_dst[rows, leader]-12
    roots = torch.zeros(3, 4, 3)
    boxes = torch.zeros(3, 8, 13)
    sizes = torch.ones(3, 8, 3)*.4
    goals = torch.zeros(3, 4, 3)
    goals[..., 0] = torch.arange(4)*3.
    source = g.edge_src[rows, edges]
    if family == 'stack':
        boxes[rows, source-4, :2] = goals[rows, goal, :2]
    else:
        roots[rows, source, :2] = goals[rows, goal, :2]
    raw = torch.zeros(3, len(g.ids), dtype=torch.bool)
    raw[rows[::2], leader[::2]] = True
    original = torch.rand(3, len(g.ids))
    args = (roots, boxes, sizes, goals, g, CONFIG)
    actual = predecessor_at_progress(original, *args, raw_success=raw, lambda_inner=10.)
    radius = sizes[rows, g.edge_dst[rows, edges]-4, :2].norm(dim=-1)/2 + (.1 if family == 'stack' else .3)
    expected = 1/(1+10*(~raw[rows, leader]).float()*radius)
    assert len(rows) == 6
    torch.testing.assert_close(actual[rows, edges], expected)
    order = torch.rand(3, len(g.ids)).argsort(-1)
    shuffled = predecessor_at_progress(original.gather(1, order), roots, boxes, sizes, goals,
        permute_graph(g, order), CONFIG, raw_success=raw.gather(1, order), lambda_inner=10.)
    assert torch.equal(shuffled, actual.gather(1, order))
    ids = torch.tensor([2, 0])
    partial = predecessor_at_progress(original[ids], roots[ids], boxes[ids], sizes[ids], goals[ids],
        select_graph(g, ids), CONFIG, raw_success=raw[ids], lambda_inner=10.)
    assert torch.equal(partial, actual[ids])


@pytest.mark.parametrize('preset', ['all_solo', 'push', 'coupled'])
def test_independent_and_coupled_progress_are_unchanged(preset):
    if preset == 'coupled':
        from utils.coupled_push_spec import compose_graph
        g = compose_graph(torch.full((3, 4), 5), torch.zeros(3, 4, dtype=torch.long), False)
    else:
        g, _, _ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
            ENV['relationGraph']['template_probabilities'], preset)
    original = torch.rand(3, len(g.ids))
    actual = predecessor_at_progress(original, torch.randn(3, 4, 3), torch.randn(3, 8, 13),
        torch.ones(3, 8, 3), torch.randn(3, 4, 3), g, CONFIG,
        raw_success=torch.zeros_like(original, dtype=torch.bool), lambda_inner=10.)
    assert torch.equal(actual, original)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_three_meter_wait_radius_only_before_gate_opens(family):
    args, rows, edges, _ = geometry(family, [0., 0., 0., 0.])
    roots, boxes, sizes, goals, graph, config = args
    source = graph.edge_src[rows, edges]
    positions = torch.tensor([[4., 0.], [3., 0.], [2.9, 0.], [2., 0.]])
    if family == 'stack':
        boxes[rows, source-4, :2] = positions
    else:
        roots[rows, source, :2] = positions
    old = torch.full((4, len(graph.ids)), .4)
    closed = torch.zeros_like(old, dtype=torch.bool)
    actual = predecessor_at_progress(old, *args, raw_success=closed, lambda_inner=10., wait_radius_m=3.)
    torch.testing.assert_close(actual[rows, edges], torch.tensor([.5, 1., .5, 1/11.]))
    opened = predecessor_at_progress(old, *args, raw_success=~closed, lambda_inner=10., wait_radius_m=3.)
    assert torch.equal(opened, predecessor_at_progress(old, *args))
    mask = ~graph.prereq_mask.any(-1)
    assert torch.equal(actual[mask], old[mask])


@pytest.mark.parametrize('radius', [0, -1, True, '3', None, float('nan'), float('inf')])
def test_invalid_wait_radius_rejected(radius):
    env = deepcopy(ENV)
    env['before']['asymmetric_progress']['wait_radius_m'] = radius
    with pytest.raises(ValueError, match='Asymmetric progress'):
        validate_unified_env(env)


def test_radius3m_experiment_contract():
    name = 'stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute'
    original, modified = native_config(name), native_config(name+'_radius3m')
    validate_unified_env(modified['env'])
    validate_experiment(name+'_radius3m', modified['env'], native_config(name+'_radius3m', True))
    assert 'approach_timeout' not in modified['env']['before']
    restored = deepcopy(modified['env'])
    assert restored['before']['asymmetric_progress'].pop('wait_radius_m') == 3.
    assert restored == original['env']
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(original['env'])}, modified['env'])
