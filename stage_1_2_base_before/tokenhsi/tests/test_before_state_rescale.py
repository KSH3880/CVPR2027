from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from utils.experiments import native_config, validate_experiment
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.unified_training import validate_unified_env
from test_before_atprogress import fixture

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute'
NAME = BASE + '_state_rescale'
ENV = native_config(NAME)['env']


def setup(family, enabled=True):
    g, _, _, _, _, rows, edges, _, _ = fixture(family)
    rt = BeforePushRuntime(4, g, ENV['relationReward'], 'cpu',
        contact_weights=ENV['jointPush']['contact_weights'], state_rescale=enabled)
    phi = torch.full_like(rt.phi, .2)
    error = torch.ones_like(phi)
    leader = g.prereq_mask[rows, edges].long().argmax(-1)
    return rt, phi, error, rows, edges, leader


def tick(rt, phi, error):
    return rt.step(phi, torch.full_like(phi, .4), error, error, error)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_open_stationary_improve_regress_max_and_original_components(family):
    rt, phi, err, rows, edges, leader = setup(family)
    assert (tick(rt, phi, err)['state_component'][rows, edges] == 0).all()
    assert not rt.state_open_seen.any()
    phi[rows, leader] = 1; err[rows, leader] = 0
    phi[rows, edges] = torch.tensor([.2, .5, .9, .999999])
    anchor = phi[rows, edges].clone()
    for _ in range(3):
        result = tick(rt, phi, err)
        assert (result['state_component'][rows, edges] == 0).all()
    # Feed the original nonlinear phi values: only affine output mapping changes.
    for fraction in [.25, .75, .1, 1.]:
        phi[rows, edges] = anchor + fraction*(1-anchor)
        result = tick(rt, phi, err)
        expected = .2*((phi[rows, edges]-anchor)/(1-anchor)).clamp(0, 1)
        torch.testing.assert_close(result['state_component'][rows, edges], expected)
        original = rt.components(phi, torch.full_like(phi, .4), err, err, err, rt.graph)
        for key in ['phi_raw', 'progress_component', 'success_component', 'own_success']:
            assert torch.equal(result[key], original[key])
        independent = ~rt.graph.prereq_mask.any(-1)
        assert torch.equal(result['state_component'][independent], original['state_component'][independent])
        torch.testing.assert_close(result['total'], sum(result[k] for k in ['state_component','progress_component','success_component']))
        from env.tasks.multi_agent.edge_context_reward import owner_sum
        torch.testing.assert_close(result['local_task_reward'], owner_sum(result['total'], rt.graph))
    assert (result['state_component'][rows, edges] == .2).all()
    phi[rows, edges] = anchor/2
    assert (tick(rt, phi, err)['state_component'][rows, edges] == 0).all()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_relock_preserves_anchor_and_partial_reset_clears_only_selected_rows(family):
    rt, phi, err, rows, edges, leader = setup(family)
    phi[rows, leader] = 1; err[rows, leader] = 0
    tick(rt, phi, err)
    saved = rt.state_open_value.clone()
    phi[rows, leader] = .1
    phi[rows, edges] = .6
    assert (tick(rt, phi, err)['state_component'][rows, edges] == 0).all()
    phi[rows, leader] = 1
    torch.testing.assert_close(tick(rt, phi, err)['state_component'][rows, edges], torch.full((4,), .1))
    assert torch.equal(saved, rt.state_open_value)
    ids = torch.tensor([3, 1])
    rt.reset(ids, phi[ids], err[ids], err[ids], err[ids])
    assert not rt.state_open_seen[ids].any()
    assert torch.equal(rt.state_open_value[[0,2]], saved[[0,2]])
    result = tick(rt, phi, err)
    assert (result['state_component'][ids, edges[ids]] == 0).all()
    torch.testing.assert_close(result['state_component'][[0,2], edges[[0,2]]], torch.full((2,), .1))


def test_already_max_is_finite_and_keeps_max_without_delayed_time_ramp():
    rt, phi, err, rows, edges, leader = setup('sit')
    phi.fill_(1); err.zero_()
    for _ in range(2):
        result = tick(rt, phi, err)
        assert torch.isfinite(result['total']).all()
        assert (result['state_component'][rows, edges] == .2).all()


def test_disabled_matches_original_and_components_do_not_commit():
    rt, phi, err, rows, edges, leader = setup('sit', False)
    expected = rt.components(phi, phi, err, err, err, rt.graph)
    actual = rt.step(phi, phi, err, err, err)
    for key in expected: assert torch.equal(actual[key], expected[key])
    rt, phi, err, rows, edges, leader = setup('sit')
    phi[rows, leader]=1;err[rows, leader]=0
    rt.components(phi, phi, err, err, err, rt.graph)
    assert not rt.state_open_seen.any()


@pytest.mark.parametrize('base_name', [BASE, BASE.replace('_virtual', ''), BASE.replace('_virtual', '_virtual_asym')])
def test_config_separate_and_old_env_unchanged(base_name):
    name = base_name + '_state_rescale'
    base, cfg = native_config(base_name), native_config(name)
    validate_unified_env(cfg['env'])
    validate_experiment(name, cfg['env'], native_config(name, True))
    env = deepcopy(cfg['env'])
    assert env['before'].pop('state_rescale') == 'first_open_affine_v1'
    assert env == base['env'] and cfg['sim'] == base['sim']
    if '_virtual' not in base_name:
        assert 'before_virtual_object' not in env['relationGraph']
        assert 'wait_clear' not in env['before'] and 'asymmetric_progress' not in env['before']
    assert cfg['experiment']['transfer'] == base['experiment']['transfer']
    for source, target in [(base,cfg),(cfg,base)]:
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract':checkpoint_contract(source['env'])},target['env'])
    for value in [None, True, 'unknown']:
        bad=deepcopy(cfg['env']);bad['before']['state_rescale']=value
        with pytest.raises(ValueError, match='state_rescale'): validate_unified_env(bad)
