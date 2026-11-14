import pytest
import torch

from env.tasks.multi_agent.before_wait import BeforeWait
from test_before_wait import setup
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2'
NAME = BASE[:-1] + '3'
OPTION = native_config(NAME)['env']['before']['waiting_penalty']


def test_config_only_changes_radius_and_experiment_name():
    cfg, base = native_config(NAME), native_config(BASE)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert OPTION == {'radius_buffer': 1.5, 'coefficient': 1., 'ontop_distance_source': 'human'}
    check_checkpoint({'before_contract': checkpoint_contract(cfg['env'])}, cfg['env'])
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(base['env'])}, cfg['env'])
    cfg['env']['before']['waiting_penalty']['radius_buffer'] = 2.5
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_smaller_root_boundary_keeps_strength_and_ever_release(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    center = goals[rows, state.goal, :2]
    radius = sizes[rows, state.support, :2].norm(dim=-1)/2
    roots[rows, state.owners, :2] = center
    roots[rows, state.owners, 0] += radius + 2.
    assert not state.penalty(*args).any()
    legacy = BeforeWait(graph, len(roots), .1, native_config(BASE)['env']['before']['waiting_penalty'], 'cpu')
    legacy.reset(torch.arange(len(roots)), graph, raw)
    assert legacy.penalty(*args).sum() > 0
    roots[rows, state.owners, 0] -= .51
    velocity.zero_(); velocity[..., 2] = .5
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .5
    torch.testing.assert_close(state.penalty(*args), expected)
    roots[rows, state.owners, :2] += 100
    torch.testing.assert_close(state.penalty(*args), expected)
    raw[rows, state.predecessor] = True
    assert not state.penalty(*args).any()
    raw.zero_()
    assert not state.penalty(*args).any()
