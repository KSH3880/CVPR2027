from copy import deepcopy
import math

import pytest
import torch

from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.edge_ontop_spec import permute_graph, select_graph
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from test_before_asymmetric_progress import geometry

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute'
NAME = BASE + '_hardgate'
ENV = native_config(NAME)['env']


def test_configuration_and_checkpoint_separation():
    cfg, base = native_config(NAME), native_config(BASE)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    restored = deepcopy(cfg['env'])
    assert restored['before'].pop('hard_gate_progress') is True
    assert restored == base['env']
    assert cfg['sim'] == base['sim']
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)
    for source, target in ((cfg, base), (base, cfg)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source['env'])}, target['env'])


@pytest.mark.parametrize('option', [False, None, 1, 'true', {}])
def test_invalid_option(option):
    env = deepcopy(ENV)
    env['before']['hard_gate_progress'] = option
    with pytest.raises(ValueError, match='Hard gate progress'):
        validate_unified_env(env)


def test_requires_virtual_and_disallows_asym():
    for change in ('virtual', 'asym'):
        env = deepcopy(ENV)
        if change == 'virtual':
            env['relationGraph'].pop('before_virtual_object')
        else:
            env['before']['asymmetric_progress'] = {'lambda_inner': 10.}
        with pytest.raises(ValueError, match='Hard gate progress'):
            validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_boundary_and_current_frame_recovery(family):
    args, rows, edges, _ = geometry(family, [.1, 0., -.05, -.2])
    graph, config = args[-2:]
    original = torch.rand(4, len(graph.ids))
    saved = original.clone()
    ordinary = predecessor_at_progress(original, *args)
    runtime = BeforePushRuntime(4, graph, config, 'cpu', contact_weights=ENV['jointPush']['contact_weights'])
    leader = graph.prereq_mask[rows, edges].long().argmax(-1)
    # Historical success must not override the current-frame prerequisite.
    runtime.achieved.fill_(True)
    runtime.own_success.fill_(True)
    for success in (False, True, False):
        phi = torch.full_like(original, .1)
        phi[rows, leader] = math.exp(-10*(.1 if success else .3)**2)
        errors = torch.ones_like(phi)
        z = errors.clone()
        z[rows, leader] = 0.
        raw = interaction_own_success(phi, z, errors, graph, config, errors)
        actual = predecessor_at_progress(original, *args, raw_success=raw, hard_gate=True)
        expected = ordinary.clone()
        if not success:
            expected[rows[1:], edges[1:]] = 0.
        assert torch.equal(actual, expected)
        new = runtime.components(phi, actual, z, errors, errors, graph)
        old = runtime.components(phi, ordinary, z, errors, errors, graph)
        torch.testing.assert_close(new['progress_component'][rows, edges], expected[rows, edges]*config['progress_reward_weight'])
        for key in ('state_component', 'success_component', 'own_success', 'reward_saturated'):
            assert torch.equal(new[key], old[key])
        assert torch.equal(actual[~graph.prereq_mask.any(-1)], original[~graph.prereq_mask.any(-1)])
        order = torch.rand_like(original).argsort(-1)
        shuffled = predecessor_at_progress(original.gather(1, order), *args[:4], permute_graph(graph, order), config,
            raw_success=raw.gather(1, order), hard_gate=True)
        assert torch.equal(shuffled, actual.gather(1, order))
        ids = torch.tensor([3, 1])
        partial = predecessor_at_progress(original[ids], *(a[ids] for a in args[:4]), select_graph(graph, ids), config,
            raw_success=raw[ids], hard_gate=True)
        assert torch.equal(partial, actual[ids])
    assert torch.equal(original, saved)
