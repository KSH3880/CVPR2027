from copy import deepcopy

import pytest
import torch

from test_before_wait import NAME as OLD, setup
from utils.experiments import native_config, validate_experiment, NAMES
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.stage1_jointpush import compose_graph
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges

NAME = OLD + '_v2'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_termination']


def test_config_changes_only_waiting_and_at_height_constraint():
    cfg, old = native_config(NAME), native_config(OLD)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg)
    restored['env']['before']['waiting_termination'] = old['env']['before']['waiting_termination']
    restored['env']['relationReward']['success']['at_z_tolerance'] = .001
    restored['experiment']['name'] = OLD
    assert restored == old
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(OLD, True)['params']['config']['name']
    assert train == native_config(OLD, True)
    for source, target in ((old, cfg), (cfg, old)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])
    for name in NAMES:
        validate_experiment(name, native_config(name)['env'], native_config(name, True))


@pytest.mark.parametrize('key,value', [('speed_axes', 'z'), ('speed_axes', None),
    ('violation_mode', 'consecutive'), ('violation_mode', None), ('extra', True)])
def test_invalid_modes(key, value):
    env = deepcopy(ENV)
    env['before']['waiting_termination'][key] = value
    with pytest.raises(ValueError, match='waiting termination'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
@pytest.mark.parametrize('velocity', [(0, 0, .31), (0, 0, -.31), (.2, .2, .2)])
def test_immediate_xyz_cumulative_time_and_permanent_arming(family, velocity):
    state, args, graph, rows, edges = setup(family, dt=1/30, config=OPTION)
    args[1][:] = torch.tensor(velocity)
    snapshots = [v.clone() for v in args]
    assert state.grace_steps == 0 and state.violation_steps == 30
    assert not state.step(*args).any()
    assert (state.moving_steps[rows, edges] == 1).all()
    assert all(torch.equal(a, b) for a, b in zip(args, snapshots))
    # Leaving the radius and pausing must not erase previously accumulated time.
    args[0][..., :2] += 100
    args[2][..., :2] += 100
    for expected in range(2, 30):
        args[1].zero_()
        for _ in range(3):
            assert not state.step(*args).any()
            assert (state.moving_steps[rows, edges] == expected-1).all()
        args[1][:] = torch.tensor(velocity)
        assert not state.step(*args).any()
    result = state.step(*args)
    expected = torch.zeros_like(result)
    expected[rows, graph.edge_owner[rows, edges]] = True
    assert torch.equal(result, expected)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_xy_entry_radius_and_speed_threshold(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    entity = boxes if family == 'stack' else roots
    index = state.source-4 if family == 'stack' else state.owners
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2 + 1.
    entity[rows, index, :2] = goals[rows, state.goal, :2]
    entity[rows, index, 0] += radius + .001
    for _ in range(15):
        assert not state.step(*args).any()
    assert not state.armed.any()
    entity[rows, index, 0] -= .002
    goals[..., 2] += 100
    velocity.zero_(); velocity[..., 2] = .3
    for _ in range(15):
        assert not state.step(*args).any()
    assert state.armed[rows, edges].all() and not state.moving_steps.any()
    velocity[..., 2] = .301
    state.step(*args)
    assert (state.moving_steps[rows, edges] == 1).all()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_real_at_predicate_gates_and_releases_on_termination_frame(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    runtime = BeforePushRuntime(len(roots), graph, ENV['relationReward'], 'cpu', contact_weights=[.2]*3)
    leader = state.predecessor
    payload = graph.edge_src[rows, leader]-4
    for _ in range(state.violation_steps-1):
        assert not state.step(*args).any()
    boxes[rows, payload, :3] = goals[rows, state.goal]+torch.tensor([.1, 0, .1])
    hands = roots[:, :, None].expand(-1, -1, 2, -1)
    phi, diag = evaluate_interaction_edges(hands, hands, roots, boxes, sizes, goals, graph, ENV['relationReward'], .9)
    result = runtime.components(phi, diag['progress'], diag['z_error'],
        diag['feet_height_error'], diag['region_error'], graph)
    raw.copy_(result['raw_own_success'])
    assert raw[rows, leader].all() and result['prerequisite_gate'][rows, edges].all()
    assert not state.step(*args).any()
    raw.zero_()
    for _ in range(25):
        assert not state.step(*args).any()
    state.reset(torch.tensor([1]), graph, raw)
    assert state.released[rows[rows != 1], edges[rows != 1]].all()
    for _ in range(state.violation_steps-1):
        assert not state.step(*args).any()
    result = state.step(*args)
    assert result.sum() == 1 and result[1].any()


@pytest.mark.parametrize('template', [3, 5])
@pytest.mark.parametrize('delta,success', [([.1, 0, .1], True), ([0, 0, .199], True),
    ([.199, 0, 0], True), ([.15, 0, .15], False), ([0, 0, .201], False), ([.201, 0, 0], False)])
def test_solo_at_and_push_at_use_same_3d_distance(template, delta, success):
    graph = compose_graph(torch.full((2, 4), template), shuffle=True)
    roots = torch.zeros(2, 4, 3)
    hands = roots[:, :, None].expand(-1, -1, 2, -1)
    boxes = torch.zeros(2, 8, 13); boxes[..., 6] = 1
    sizes, goals = torch.full((2, 8, 3), .4), torch.zeros(2, 4, 3)
    rows, edges = ((graph.edge_relation == 7) & graph.edge_valid).nonzero(as_tuple=True)
    boxes[rows, graph.edge_src[rows, edges]-4, :3] = torch.tensor(delta)
    phi, diag = evaluate_interaction_edges(hands, hands, roots, boxes, sizes, goals, graph, ENV['relationReward'], .9)
    runtime = BeforePushRuntime(2, graph, ENV['relationReward'], 'cpu', contact_weights=[.2]*3)
    result = runtime.components(phi, diag['progress'], diag['z_error'],
        diag['feet_height_error'], diag['region_error'], graph)
    assert len(rows) == 8 and (result['raw_own_success'][rows, edges] == success).all()
