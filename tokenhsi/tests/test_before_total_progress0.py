from copy import deepcopy

import pytest
import torch

from test_before_atprogress import fixture
from env.tasks.multi_agent.before_reward import BeforeRuntime, predecessor_at_progress
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.edge_ontop_spec import permute_graph
from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.unified_training import validate_unified_env

NAME = 'stage2_before_total_progress0'
ENV = native_config(NAME)['env']


def test_only_progress_radius_and_experiment_names_change():
    cfg, base = native_config(NAME), native_config('stage2_before_total')
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    assert cfg['env']['before'].pop('progress_radius') == 0.
    cfg['experiment']['name'] = base['experiment']['name']
    assert cfg == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = 'Stage2BeforeTotal'
    assert train == native_config('stage2_before_total', True)
    with pytest.raises(ValueError, match='contract differs'):
        check_checkpoint({'before_contract': checkpoint_contract(base['env'])}, ENV)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, None])
def test_invalid_radius_rejected(value):
    env = deepcopy(ENV)
    env['before']['progress_radius'] = value
    with pytest.raises(ValueError, match='Progress radius'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_center_progress_preserves_gate_success_holding_and_graph_mapping(family):
    g, roots, boxes, sizes, goals, rows, edges, goal_ids, _ = fixture(family)
    old = torch.full(g.edge_relation.shape, .4)
    source = boxes[rows, g.edge_src[rows, edges]-4, :2] if family == 'stack' else roots[rows, g.edge_src[rows, edges], :2]
    distance = torch.tensor([0., .1, .5, 5.])
    goals[rows, goal_ids, :2] = source
    goals[rows, goal_ids, 0] += distance
    args = roots, boxes, sizes, goals, g, ENV['relationReward']
    progress = predecessor_at_progress(old, *args, radius_override=0.)
    torch.testing.assert_close(progress[rows, edges], 1/(1+distance))
    assert torch.equal(progress[~g.prereq_mask.any(-1)], old[~g.prereq_mask.any(-1)])
    assert torch.equal(old, torch.full_like(old, .4))
    order = torch.rand_like(old).argsort(-1)
    shuffled = predecessor_at_progress(old.gather(1, order), roots, boxes, sizes, goals,
        permute_graph(g, order), ENV['relationReward'], radius_override=0.)
    torch.testing.assert_close(shuffled, progress.gather(1, order))
    runtime = BeforeRuntime(4, g, ENV['relationReward'], 'cpu')
    zero = torch.zeros_like(old)
    for leader_success in (False, True, False):
        phi = torch.full_like(old, .1)
        phi[g.edge_relation == 7] = 1. if leader_success else .1
        failed = torch.ones_like(old)
        failed[g.edge_relation == 7] = 0
        a = runtime.components(phi, progress, failed, failed, failed, g)
        b = runtime.components(phi, old, failed, failed, failed, g)
        for key in ('prerequisite_gate', 'state_component', 'success_component', 'own_success'):
            assert torch.equal(a[key], b[key])
        torch.testing.assert_close(a['progress_component'][rows, edges], .2/(1+distance))
    success = runtime.components(torch.ones_like(old), progress, zero, zero, zero, g)
    torch.testing.assert_close(success['total'][rows, edges], torch.full((4,), .6))
