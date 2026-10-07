"""Pure CPU tensor fixtures, including known (not solved) reward exploits."""
import pytest
import torch
from utils.relation_task_spec import compile_carry_subgoal, validate_relation_config, STATE_MODE
from env.tasks.multi_agent.relation_reward import (
    evaluate_holding, evaluate_at, relation_gate,
    prerequisite_minimum, pin_progress, relation_step, box_speed_penalty)






def test_state_evaluators_boundaries_and_proxy_limitation():
    obj = torch.zeros(1, 3)
    # Opposite hands far from box are a documented midpoint false positive.
    hands = torch.tensor([[[-2., 0., 0.], [2., 0., 0.]]])
    assert evaluate_holding(hands, obj)[0].item() == 1
    assert evaluate_holding(hands + 1, obj)[0].item() < 1
    assert evaluate_at(obj, obj)[0].item() == 1
    near_z = torch.tensor([[0., 0., .00099]])
    far_z = torch.tensor([[0., 0., .00101]])
    assert evaluate_at(obj, near_z)[2].item()
    assert not evaluate_at(obj, far_z)[2].item()
    # Placement is a separate diagnostic; crossing Z tolerance does not jump phi.
    assert evaluate_at(obj, far_z)[0].item() > .9
    assert abs(evaluate_at(obj, near_z)[0].item() - evaluate_at(obj, far_z)[0].item()) < 1e-5
    assert evaluate_at(obj, torch.tensor([[.09999, 0., 0.]]))[2].item()
    assert not evaluate_at(obj, torch.tensor([[.10001, 0., 0.]]))[2].item()


def test_box_near_at_state_is_only_the_xyz_gaussian():
    obj = torch.zeros(2, 3)
    goal = torch.tensor([[.1, 0., 0.], [0., 0., .00101]])
    state, near, put, _, _ = evaluate_at(obj, goal, state_definition='box_near')
    torch.testing.assert_close(state, near)
    torch.testing.assert_close(state, torch.exp(-10 * goal.square().sum(-1)))
    assert put.tolist() == [True, False]


def test_box_near_state_threshold_drives_success_without_legacy_putdown():
    obj = torch.zeros(1, 3)
    goal = torch.tensor([[0., 0., .05]])
    at, _, legacy_put, _, _ = evaluate_at(obj, goal, state_definition='box_near')
    assert at.item() > .9
    assert not legacy_put.item()
    graph = compile_carry_subgoal(1, 1)
    before = torch.tensor([[1., 0.]])
    after = torch.stack([torch.ones_like(at), at], -1)
    achieved = torch.tensor([[True, False]])
    result = relation_step(before, after, torch.zeros_like(after), achieved,
                           torch.zeros(1, 1, dtype=torch.bool), graph, success_bonus=10.)
    assert result['first_success'].item()
    assert result['success_bonus'].item() == 10.


def test_gate_and_prerequisite_minimum():
    phi = torch.tensor([[0., .8, .9, 1.]])
    gate = relation_gate(phi)
    assert (gate[:, 1:] > gate[:, :-1]).all()
    assert gate[0, 1].item() == .5
    assert gate[0, 2].item() == pytest.approx(.952574, abs=1e-6)
    pre = torch.zeros(4, 4, dtype=torch.bool)
    pre[3, :3] = True
    result = prerequisite_minimum(gate, pre)
    assert (result[0, :3] == 1).all()
    assert result[0, 3].item() == pytest.approx(gate[0, :3].min().item())


def test_distance_state_progress_and_no_mutation():
    graph = compile_carry_subgoal(1, 1)
    before = torch.tensor([[.8, .2]])
    after = torch.tensor([[.9, .4]])
    progress = torch.tensor([[.3, .6]])
    achieved = torch.zeros(1, 2, dtype=torch.bool)
    done = torch.zeros(1, 1, dtype=torch.bool)
    result = relation_step(before, after, progress, achieved, done, graph)
    gate = relation_gate(before)
    activation = torch.tensor([[1., gate[0, 0]]])
    pinned = progress
    torch.testing.assert_close(result['activation'], activation)
    torch.testing.assert_close(result['pinned_progress'], pinned)
    torch.testing.assert_close(result['state_component'], .2 * activation * after)
    torch.testing.assert_close(result['progress_component'], .2 * activation * pinned)
    assert not achieved.any() and not done.any()


def test_progress_pinning_endpoints():
    progress = torch.tensor([[0., .25, 1.]])
    gate = torch.tensor([[0., .5, 1.]])
    torch.testing.assert_close(pin_progress(progress, gate), torch.tensor([[0., .625, 1.]]))


def test_done_only_latches_success_bonus_not_dense_reward():
    graph = compile_carry_subgoal(1, 1)
    phi = torch.ones(1, 2)
    achieved = torch.ones(1, 2, dtype=torch.bool)
    done = torch.ones(1, 1, dtype=torch.bool)
    result = relation_step(phi, phi, torch.zeros_like(phi), achieved, done, graph)
    assert result['edge_reward'].sum() > 0
    assert result['agent_task_reward'].item() > 0
    assert result['success_bonus'].item() == 0
    assert result['done_next'].item()


def test_separate_box_penalty_and_config_validation():
    x = torch.zeros(2, 3)
    y = torch.tensor([[.1, 0., 0.], [.3, 0., 0.]])
    value = box_speed_penalty(x, y, .1)
    assert value[0] == 0 and value[1] < 0
    validate_relation_config({'mode': STATE_MODE})
    validate_relation_config({'mode': STATE_MODE,
                              'at': {'state_definition': 'box_near', 'near_distance_scale': 10.}})
    for cfg in ({'mode': 'push'}, {'mode': STATE_MODE, 'state_reward_weight': float('nan')},
                {'mode': STATE_MODE, 'success': {'terminate_when_all_subgoals_done': True}},
                {'mode': STATE_MODE, 'observation': {'include_subgoal_done': False}}):
        with pytest.raises(ValueError):
            validate_relation_config(cfg)
    for at in ({'state_definition': 'unknown'},
               {'state_definition': 'box_near', 'near_fraction': .5},
               {'state_definition': 'box_near', 'putdown_xy_tolerance': .1}):
        with pytest.raises(ValueError):
            validate_relation_config({'mode': STATE_MODE, 'at': at})


def test_input_contract_and_unsupported_operator_fail_loudly():
    graph = compile_carry_subgoal(1, 1)
    phi = torch.zeros(1, 2)
    a, done = torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
    for invalid in (torch.tensor([[float('nan'), 0.]]), torch.tensor([[1.1, 0.]])):
        with pytest.raises(ValueError, match='finite'):
            relation_step(phi, invalid, phi, a, done, graph)
    with pytest.raises(ValueError, match='boolean'):
        relation_step(phi, phi, phi, a.float(), done, graph)
    with pytest.raises(ValueError, match='Unsupported'):
        validate_relation_config({'mode': STATE_MODE, 'operator': 'Push'})
    with pytest.raises(ValueError, match='Unsupported'):
        validate_relation_config({'mode': STATE_MODE, 'holding': {'type': 'OnTop'}})
