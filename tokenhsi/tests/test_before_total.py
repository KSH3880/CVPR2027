from copy import deepcopy

import pytest
import torch

from test_before_wait import setup
from test_evaluation_protocol import args
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.evaluation_protocol import configure
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3_ep900'
NAME = 'stage2_before_total'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_penalty']


def test_fresh_26k_profile_only_changes_ontop_entry_and_names():
    cfg, base = native_config(NAME), native_config(BASE)
    train = native_config(NAME, True)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], deepcopy(train))
    assert OPTION == {'radius_buffer': 1.5, 'coefficient': 1., 'ontop_distance_source': 'payload'}
    assert cfg['experiment']['transfer']['checkpoint'] == 'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
    assert not train['params']['load_checkpoint']
    assert train['params']['config']['name'] == 'Stage2BeforeTotal'
    cfg['env']['before']['waiting_penalty']['ontop_distance_source'] = 'human'
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    for source, target in ((ENV, base['env']), (base['env'], ENV)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)


def test_training_and_formal_evaluation_keep_900_steps():
    cfg = native_config(NAME)
    assert cfg['env']['episodeLength'] == experiment_horizon(NAME) == 900
    cfg['env']['episodeLength'] = 1200
    with pytest.raises(ValueError, match='requires 900 steps'):
        validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    configure(args(), cfg, NAME)
    assert cfg['env']['episodeLength'] == 900
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))


@pytest.mark.parametrize('payload_near', [False, True])
def test_ontop_entry_uses_payload_but_penalty_uses_human_xyz_speed(payload_near):
    state, inputs, graph, rows, edges = setup('stack', config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = inputs
    center = goals[rows, state.goal, :2]
    roots[rows, state.owners, :2] = center + (20 if payload_near else 0)
    boxes[rows, state.source - roots.shape[1], :2] = center + (0 if payload_near else 20)
    velocity.zero_()
    velocity[..., 2] = .7
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .7 if payload_near else 0
    torch.testing.assert_close(state.penalty(*inputs), expected)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_xy_boundary_ever_entry_ever_release_and_partial_reset(family):
    state, inputs, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = inputs
    center = goals[rows, state.goal, :2].clone()
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2 + 1.5
    locations = boxes if family == 'stack' else roots
    ids = state.source - roots.shape[1] if family == 'stack' else state.owners
    locations[rows, ids, :2] = center
    locations[rows, ids, 0] += radius + .01
    assert not state.penalty(*inputs).any()
    locations[rows, ids, 0] -= .02
    goals[..., 2] += 100
    velocity[:] = torch.tensor([.3, 0., .4])
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .5
    snapshots = [v.clone() for v in inputs]
    torch.testing.assert_close(state.penalty(*inputs), expected)
    assert all(torch.equal(a, b) for a, b in zip(inputs, snapshots))
    locations[rows, ids, :2] += 100
    torch.testing.assert_close(state.penalty(*inputs), expected)
    raw[rows[rows == 0], state.predecessor[rows == 0]] = True
    expected[0] = 0
    torch.testing.assert_close(state.penalty(*inputs), expected)
    raw.zero_()
    torch.testing.assert_close(state.penalty(*inputs), expected)
    state.reset(torch.tensor([1]), graph, raw)
    expected[1] = 0
    torch.testing.assert_close(state.penalty(*inputs), expected)
    assert not state.age.any() and not state.moving_steps.any()
