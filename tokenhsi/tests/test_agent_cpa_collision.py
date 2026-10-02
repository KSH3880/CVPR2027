"""CPA geometry, reward replacement and static/CPA checkpoint isolation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.collision_reward import (
    agent_collision_config, check_agent_collision_checkpoint,
    compute_agent_collision_penalty, compute_agent_cpa_collision_penalty)
from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from utils.edge_stage2_spec import sample_graph
from utils.unified_training import validate_unified_env

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_rescue_shared9.yaml').read_text())
CPA = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_rescue_shared9_cpa.yaml').read_text())


def test_direct_approach_is_penalized_before_static_threshold_with_step_discount():
    pos = torch.tensor([[[0., 0., 0.], [2., 0., 3.]]])
    vel = torch.tensor([[[1., 0., 10.], [-1., 0., -10.]]])
    actual = compute_agent_cpa_collision_penalty(pos, vel, .7, .99, 1/30)
    torch.testing.assert_close(actual, torch.full((1, 2), .99 ** 30))
    assert not compute_agent_collision_penalty(pos, .7).any()
    # A control step of 1/60 discounts the same one-second encounter twice as much.
    torch.testing.assert_close(compute_agent_cpa_collision_penalty(pos, vel, .7, .99, 1/60),
                               torch.full((1, 2), .99 ** 60))


@pytest.mark.parametrize('kind', ['stationary', 'receding', 'parallel', 'miss', 'coincident'])
def test_no_cpa_risk_for_non_closing_or_safe_passes_and_stationary_overlap(kind):
    pos = torch.tensor([[[0., 0., 0.], [.2, 0., 0.]]])
    vel = torch.zeros_like(pos)
    if kind == 'receding':
        vel[0, 0, 0] = -1
    elif kind == 'parallel':
        vel[..., 1] = 1
    elif kind == 'miss':
        pos[0, 1] = torch.tensor([2., 1., 0.]); vel[0, 0, 0] = 1
    elif kind == 'coincident':
        pos.zero_(); vel[0, 0, 0] = 1
    result = compute_agent_cpa_collision_penalty(pos, vel, .7, .99, 1/30)
    assert torch.isfinite(result).all() and not result.any()
    if kind == 'stationary':
        assert (compute_agent_collision_penalty(pos, .7) > 0).all()


def test_cpa_is_bounded_symmetric_and_invariant_to_translation_velocity_and_human_order():
    torch.manual_seed(7)
    pos = torch.randn(16, 4, 3); vel = torch.randn_like(pos)
    risk = compute_agent_cpa_collision_penalty(pos, vel, .7, .99, 1/30)
    assert torch.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all()
    order = torch.tensor([2, 0, 3, 1])
    shuffled = compute_agent_cpa_collision_penalty(pos[:, order], vel[:, order], .7, .99, 1/30)
    torch.testing.assert_close(shuffled, risk[:, order])
    shifted = compute_agent_cpa_collision_penalty(pos + torch.tensor([8., -5., 20.]),
        vel + torch.tensor([4., 3., -11.]), .7, .99, 1/30)
    torch.testing.assert_close(shifted, risk, atol=2e-6, rtol=2e-5)
    for i in range(4):
        pairs = [compute_agent_cpa_collision_penalty(pos[:, [i, j]], vel[:, [i, j]],
                 .7, .99, 1/30)[:, 0] for j in range(4) if j != i]
        torch.testing.assert_close(risk[:, i], torch.stack(pairs, -1).max(-1).values)
    assert not compute_agent_cpa_collision_penalty(pos[:, :1], vel[:, :1], .7, .99, 1/30).any()


def test_static_computation_preserves_previous_distance_only_behavior():
    pos = torch.tensor([[[0., 0., 0.], [.35, 0., 8.], [2., 0., 0.]]])
    torch.testing.assert_close(compute_agent_collision_penalty(pos, .7), torch.tensor([[.5, .5, 0.]]))


def test_new_config_changes_only_experiment_name_and_agent_cpa_settings():
    validate_unified_env(CPA['env'])
    assert CPA['env']['numEnvs'] == 2048
    env = deepcopy(CPA['env'])
    assert env.pop('agentCollisionMode') == 'cpa'
    assert env.pop('agentCollisionTTCDiscount') == .99
    assert env == BASE['env']
    assert agent_collision_config(CPA['env']) == dict(
        mode='cpa', enabled=True, coefficient=.5, distance=.7, ttc_discount=.99)
    assert agent_collision_config(BASE['env'])['mode'] == 'static'
    assert not any('otherObjectCollision' in key for key in env)


@pytest.mark.parametrize('key,value', [('agentCollisionMode', 'bad'),
    ('agentCollisionPenalty', 1), ('agentCollisionCoeff', -1),
    ('agentCollisionCoeff', float('nan')), ('agentCollisionDist', 0),
    ('agentCollisionTTCDiscount', 0), ('agentCollisionTTCDiscount', 1.01)])
def test_invalid_collision_settings_fail_instead_of_silently_changing_rewards(key, value):
    env = deepcopy(CPA['env']); env[key] = value
    with pytest.raises(ValueError):
        agent_collision_config(env)


def test_checkpoint_contract_rejects_cross_mode_and_changed_parameters_and_reads_old_static_runs():
    cpa = {'agent_collision_config': agent_collision_config(CPA['env'])}
    static = {'relation_experiment_config': {'env': BASE['env']}}
    check_agent_collision_checkpoint(cpa, CPA['env'])
    check_agent_collision_checkpoint(static, BASE['env'])
    check_agent_collision_checkpoint({}, BASE['env'])
    for weights, env in [(static, CPA['env']), (cpa, BASE['env']), ({}, CPA['env'])]:
        with pytest.raises(ValueError, match='collision config differs'):
            check_agent_collision_checkpoint(weights, env)
    for key, value in [('agentCollisionCoeff', 1.), ('agentCollisionDist', .8),
                       ('agentCollisionTTCDiscount', .98), ('agentCollisionPenalty', False)]:
        changed = deepcopy(CPA['env']); changed[key] = value
        with pytest.raises(ValueError, match='collision config differs'):
            check_agent_collision_checkpoint(cpa, changed)


def test_edge_reward_pays_selected_cpa_once_and_preserves_collision_log_key():
    task = SimpleNamespace(num_envs=1, num_agents=2, _edge_context=True,
        _edge_interaction=False, _edge_ontop=False, _power_reward=False,
        _agent_collision_penalty=True, _agent_collision_coeff=.5, _agent_collision_dist=.7,
        _box_vel_penalty=False, _edge_steps=0, _relation_cfg={'diagnostics': {'enabled': False}},
        extras={}, rew_buf=torch.zeros(2), _reward_term_sums=torch.zeros(7), _reward_term_count=0,
        _edge_ever_goal=torch.zeros(1,2,dtype=torch.bool), _edge_ever_scene=torch.zeros(1,dtype=torch.bool))
    graph = sample_graph(1, CPA['env']['relationGraph'], preset='at_climb')
    shape = graph.edge_valid.shape
    result = {k: torch.zeros(shape) for k in ['state_component', 'progress_component', 'success_component', 'reward_saturated']}
    result['agent_task_reward'] = torch.ones(1, 2)
    task.relation_runtime = SimpleNamespace(graph=graph, done=torch.zeros(1,2,dtype=torch.bool),
        own_success=torch.zeros(shape,dtype=torch.bool), step=lambda *args: result)
    task._humanoid_root_states = torch.zeros(1,2,13)
    task._humanoid_root_states[0,1,0] = .35
    task._humanoid_root_states[0,0,7] = .5
    task._humanoid_root_states[0,1,7] = -.5
    task._box_states = torch.zeros(1,4,13)
    task._reward_box_values = lambda values: values
    task._evaluate_relations = lambda: (torch.zeros(shape), {'progress': torch.zeros(shape), 'z_error': torch.zeros(shape)})
    selected = lambda roots, distance: compute_agent_cpa_collision_penalty(
        roots, task._humanoid_root_states[...,7:10], distance, .99, 1/30)
    EdgeContextTaskMixin._compute_relation_reward(task, selected)
    expected = -.5 * (.99 ** 10.5)
    torch.testing.assert_close(task.rew_buf, torch.full((2,), 1 + expected))
    torch.testing.assert_close(task.extras['reward_terms'][:,4], torch.full((2,), expected))
    # Stationary overlap has no CPA cost: a leftover static term would fail this.
    task._humanoid_root_states[...,7:10].zero_()
    EdgeContextTaskMixin._compute_relation_reward(task, selected)
    torch.testing.assert_close(task.rew_buf, torch.ones(2))
