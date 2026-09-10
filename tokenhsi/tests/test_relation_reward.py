"""Pure CPU tensor fixtures, including known (not solved) reward exploits."""
import math
import pytest
import torch
from utils.relation_task_spec import compile_carry_subgoal, validate_relation_config, STATE_MODE
from env.tasks.multi_agent.relation_reward import (
    evaluate_holding, evaluate_at, relation_gate, velocity_progress,
    prerequisite_product, relation_step, box_speed_penalty)


@pytest.mark.parametrize('speed,expected', [(1.5, 1.), (0., 0.), (-1., 0.),
                                           (1., math.exp(-1.25)), (2., math.exp(-1.25))])
def test_velocity(speed, expected):
    prev = torch.tensor([[0., 0., 0.]])
    cur = torch.tensor([[speed * .1, 0., 0.]])
    goal = torch.tensor([[4., 0., 0.]])
    result = velocity_progress(prev, cur, goal, .1)
    torch.testing.assert_close(result, torch.tensor([expected]))
    # Global translation/yaw, and target vertical offset do not affect XY progress.
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(result, velocity_progress(prev @ rot + 2, cur @ rot + 2, goal @ rot + 2, .1))
    goal[:, 2] = 10
    torch.testing.assert_close(result, velocity_progress(prev, cur, goal, .1))


def test_velocity_vertical_zero_distance_and_moving_target():
    prev = torch.zeros(1, 3)
    cur = torch.tensor([[0., 0., 1.]])
    assert velocity_progress(prev, cur, cur, .1).item() == 0
    assert velocity_progress(prev, cur, torch.tensor([[1., 0., 1.]]), .1).item() == 0
    cur = torch.tensor([[.15, 0., 0.]])
    assert velocity_progress(prev, cur, cur + torch.tensor([[1., 0., 0.]]), .1).item() == pytest.approx(1.)
    with pytest.raises(ValueError):
        velocity_progress(prev, cur, cur, 0)


@pytest.mark.parametrize('speed,expected', [(-3., -1.), (-1.5, -1.), (-.5, -1/3),
    (0., 0.), (.3, .2), (.5, 1/3), (1.5, 1.), (3., 1.)])
def test_signed_linear_velocity(speed, expected):
    prev = torch.zeros(1, 3)
    cur = torch.tensor([[speed * .1, 0., 0.]])
    target = torch.tensor([[4., 0., 0.]])
    actual = velocity_progress(prev, cur, target, .1, mode='signed_linear')
    torch.testing.assert_close(actual, torch.tensor([expected]))
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(actual, velocity_progress(
        prev @ rot + 2, cur @ rot + 2, target @ rot + 2, .1, mode='signed_linear'))
    target[:, 2] = 10.
    torch.testing.assert_close(actual, velocity_progress(prev, cur, target, .1, mode='signed_linear'))
    # Missing mode must retain the original Gaussian definition.
    torch.testing.assert_close(velocity_progress(prev, cur, target, .1),
                               velocity_progress(prev, cur, target, .1, mode='gaussian'))


def test_signed_linear_edge_cases_and_no_distance_cutoff():
    prev = torch.zeros(1, 3)
    cur = torch.tensor([[.03, 0., 0.]])
    assert velocity_progress(prev, cur, cur, .1, mode='signed_linear').item() == 0
    for distance in [.01, .3, 5.]:
        target = cur + torch.tensor([[distance, 0., 0.]])
        assert velocity_progress(prev, cur, target, .1, mode='signed_linear').item() == pytest.approx(.2)
    # Vertical and lateral movement alone give no direction progress.
    for cur in [torch.tensor([[0., 0., .1]]), torch.tensor([[0., .1, 0.]])]:
        target = cur + torch.tensor([[1., 0., 0.]])
        assert velocity_progress(prev, cur, target, .1, mode='signed_linear').item() == 0
    with pytest.raises(ValueError, match='Unsupported'):
        velocity_progress(prev, cur, target, .1, mode='unknown')
    with pytest.raises(ValueError, match='target_speed'):
        velocity_progress(prev, cur, target, .1, target_speed=0, mode='signed_linear')


def test_signed_progress_gates_and_strict_phi_bounds():
    graph = compile_carry_subgoal(1, 1)
    phi = torch.tensor([[.8, .2]])
    a, done = torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
    progress = torch.full_like(phi, -.5)
    result = relation_step(phi, phi, progress, a, done, graph,
                           state_weight=2., progress_mode='signed_linear')
    torch.testing.assert_close(result['velocity_component'], torch.tensor([[-.05, -.05]]))
    assert result['state_component'].abs().sum() == 0
    assert result['agent_task_reward'].item() == pytest.approx(-.1)
    stopped = relation_step(phi, phi, progress, a, ~done, graph, progress_mode='signed_linear')
    assert stopped['agent_task_reward'].item() == 0
    with pytest.raises(ValueError, match='progress'):
        relation_step(phi, phi, progress, a, done, graph)  # Gaussian still rejects negatives.
    for bad in [-1.01, 1.01, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='progress'):
            relation_step(phi, phi, torch.full_like(phi, bad), a, done, graph,
                          progress_mode='signed_linear')
    for bad in [-.01, 1.01, float('nan')]:
        with pytest.raises(ValueError, match='phi'):
            relation_step(phi, torch.full_like(phi, bad), progress, a, done, graph,
                          progress_mode='signed_linear')
    with pytest.raises(ValueError, match='Unsupported'):
        relation_step(phi, phi, progress, a, done, graph, progress_mode='unknown')


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
    assert evaluate_at(obj, far_z)[0].item() < .5
    assert evaluate_at(obj, near_z)[0].item() - evaluate_at(obj, far_z)[0].item() > .49
    assert evaluate_at(obj, torch.tensor([[.09999, 0., 0.]]))[2].item()
    assert not evaluate_at(obj, torch.tensor([[.10001, 0., 0.]]))[2].item()


def test_gate_and_prerequisite_direction_product():
    phi = torch.tensor([[0., .8, .9, 1.]])
    gate = relation_gate(phi)
    assert (gate[:, 1:] > gate[:, :-1]).all()
    assert gate[0, 1].item() == .5
    assert gate[0, 2].item() == pytest.approx(.952574, abs=1e-6)
    pre = torch.zeros(4, 4, dtype=torch.bool)
    pre[3, :2] = True
    result = prerequisite_product(gate, pre)
    assert (result[0, :3] == 1).all()
    assert result[0, 3].item() == pytest.approx((gate[0, 0] * gate[0, 1]).item())


def test_pretransition_gate_signed_delta_and_no_mutation():
    graph = compile_carry_subgoal(1, 1)
    before = torch.tensor([[.8, .2]])
    after = torch.tensor([[.9, .4]])
    achieved = torch.zeros(1, 2, dtype=torch.bool)
    done = torch.zeros(1, 1, dtype=torch.bool)
    result = relation_step(before, after, torch.ones_like(before), achieved, done, graph)
    torch.testing.assert_close(result['state_component'], torch.tensor([[.1, .1]]))
    assert not achieved.any() and not done.any()
    back = relation_step(after, before, torch.zeros_like(before), achieved, done, graph)
    assert (back['state_component'] < 0).all()


def test_gate_toggle_cycle_is_positive_not_potential_based():
    graph = compile_carry_subgoal(1, 1)
    states = [(1., 0.), (1., .4), (0., .4), (0., 0.), (1., 0.)]
    rewards = []
    for a, b in zip(states, states[1:]):
        out = relation_step(torch.tensor([a]), torch.tensor([b]), torch.zeros(1, 2),
                            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool), graph)
        rewards.append(out['agent_task_reward'].item())
    assert sum(rewards) == pytest.approx(.399010956, abs=1e-6)
    assert sum(.99 ** i * r for i, r in enumerate(rewards)) > 0


def test_separate_box_penalty_and_config_validation():
    x = torch.zeros(2, 3)
    y = torch.tensor([[.1, 0., 0.], [.3, 0., 0.]])
    value = box_speed_penalty(x, y, .1)
    assert value[0] == 0 and value[1] < 0
    validate_relation_config({'mode': STATE_MODE})
    for cfg in ({'mode': 'push'}, {'mode': STATE_MODE, 'state_delta_weight': float('nan')},
                {'mode': STATE_MODE, 'success': {'terminate_when_all_subgoals_done': True}},
                {'mode': STATE_MODE, 'observation': {'include_subgoal_done': False}}):
        with pytest.raises(ValueError):
            validate_relation_config(cfg)


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
