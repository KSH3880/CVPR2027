from copy import deepcopy

import pytest
import torch

from sa_helpers import configs
from utils.edge_context_spec import AT, HOLDING
from utils.joint_carry_spec import validate_env
from utils.joint_carry_amp import validate_training
from utils.mixed_carry_spec import sample_graph
from utils.relation_task_spec import (
    validate_relation_config, checkpoint_metadata, check_checkpoint_metadata,
)
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.joint_carry_reward import apply_joint_geometry, JointCarryRuntime


@pytest.mark.parametrize('name', ['stage2', 'stage2_grasp'])
def test_at20_changes_only_success_radius_and_run_name(name):
    base, train = configs(name)
    added, new_train = configs(name + '_at20')
    validate_env(added)
    validate_training(added, new_train)
    validate_relation_config(base['relationReward'])
    validate_relation_config(added['relationReward'])
    assert added['relationReward']['success'].pop('at_distance_tolerance') == 0.2
    assert added == base
    assert new_train['params']['config'].pop('name') == name + '_at20'
    assert train['params']['config'].pop('name') == name
    assert new_train == train


@pytest.mark.parametrize('name', ['stage2', 'stage2_grasp'])
def test_checkpoint_keeps_success_contract_separate(name):
    base = checkpoint_metadata(configs(name)[0]['relationReward'])
    added = checkpoint_metadata(configs(name + '_at20')[0]['relationReward'])
    for saved, expected in [(base, base), (added, added)]:
        check_checkpoint_metadata({'relation_metadata': saved}, expected)
    for saved, expected in [(base, added), (added, base)]:
        with pytest.raises(ValueError, match='reward contract mismatch'):
            check_checkpoint_metadata({'relation_metadata': saved}, expected)
    bad = deepcopy(added['relation_reward_config'])
    bad['success']['at_distance_tolerance'] = 0.3
    with pytest.raises(ValueError, match='reward contract'):
        validate_relation_config(bad)


@pytest.mark.parametrize('name', ['stage2', 'stage2_grasp'])
@pytest.mark.parametrize('preset', ['joint_carry_at', 'independent_at_at', 'joint_carry_ontop'])
def test_at20_geometry_reset_reward_and_holding_gate(name, preset):
    env, _ = configs(name + '_at20')
    config = env['relationReward']
    offsets = torch.tensor([
        [.102, 0, 0], [.104, 0, 0], [.2, 0, 0], [.200001, 0, 0],
        [.1999, 0, .001], [.1999, 0, -.001], [.1, 0, .001001],
        [.1, 0, -.001001], [.199999, 0, .001], [0, .2, 0], [.15, .15, 0],
    ])
    n = len(offsets)
    graph = sample_graph(n, env['relationGraph'], preset=preset)
    objects = torch.zeros(n, 4, 13)
    objects[..., 6] = 1
    objects[:, :2, :3] = offsets[:, None]
    sizes = torch.tensor([.52, .8, .4]).expand(n, 4, 3)
    hands = objects[:, :2, None, :3].expand(-1, -1, 2, -1).clone()
    if preset.startswith('joint'):
        hands[:, 0, :, 1] -= .25
        hands[:, 1, :, 1] += .25
    roots = torch.zeros(n, 2, 3)
    feet = torch.zeros(n, 2, 2, 3)
    goals = torch.zeros(n, 2, 3)

    def evaluate(h):
        phi, diag = evaluate_interaction_edges(h, feet, roots, objects, sizes,
                                               goals, graph, config, .89)
        phi, gate = apply_joint_geometry(phi, diag, h, objects, sizes, graph, config)
        runtime = JointCarryRuntime(n, graph, config, 'cpu')
        runtime.coupled_holding.copy_(gate)
        args = dict(z_error=diag['z_error'], feet_error=diag['feet_height_error'],
                    region_error=diag['region_error'], distance=diag['distance'])
        runtime.reset(torch.arange(n), phi, **args)
        reset_success = runtime.own_success.clone()
        result = runtime.step(phi, diag['progress'], **args)
        torch.testing.assert_close(reset_success, result['own_success'])
        return phi, diag, result

    phi, diag, result = evaluate(hands)
    baseline = JointCarryRuntime(n, graph, configs(name)[0]['relationReward'], 'cpu')
    baseline.coupled_holding[:] = preset.startswith('joint')
    old = baseline.step(phi, diag['progress'], diag['z_error'],
                        diag['feet_height_error'], diag['region_error'])
    torch.testing.assert_close(result['state_component'], old['state_component'], rtol=0, atol=0)
    torch.testing.assert_close(result['progress_component'], old['progress_component'], rtol=0, atol=0)
    at = graph.edge_relation == AT
    holding = graph.edge_relation == HOLDING
    torch.testing.assert_close(result['own_success'][~at], old['own_success'][~at])
    if at.any():
        expected = torch.tensor([True, True, True, False, True, True, False,
                                 False, False, True, False])[:, None].expand(-1, 4)
        torch.testing.assert_close(result['own_success'][at], expected[at])
        assert not old['own_success'][at & (diag['distance'] > .104)].any()
    assert result['own_success'][holding].all()
    lost_hands = hands.clone()
    lost_hands[:, 0] += 1
    _, _, lost = evaluate(lost_hands)
    affected = graph.edge_valid & (graph.edge_relation != HOLDING)
    if preset == 'independent_at_at':
        affected &= graph.edge_owner == 0
        other = (graph.edge_relation == AT) & (graph.edge_owner == 1)
        torch.testing.assert_close(lost['own_success'][other], result['own_success'][other])
    assert not lost['own_success'][affected].any()
