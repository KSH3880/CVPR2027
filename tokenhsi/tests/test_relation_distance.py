"""Unified current-distance progress: no type, Z, motion, blending, or pinning."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

import env.tasks.multi_agent.relation_reward as reward
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import (
    compile_carry_subgoal, validate_relation_config, checkpoint_metadata, check_checkpoint_metadata)


def configs():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    names = ['approach_rsi_all_edges_success_sat', 'approach_distance_success']
    return [yaml.safe_load((directory / (name + '.yaml')).read_text()) for name in names]


def test_existing_approach_distance_config_is_unchanged():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    base = yaml.safe_load((directory / 'approach_rsi_all_edges_success_sat.yaml').read_text())
    existing = yaml.safe_load((directory / 'approach_distance.yaml').read_text())
    expected = copy.deepcopy(base)
    expected['env']['relationReward']['progress'] = {'kind': 'distance', 'delta': .5, 'sigma': 1.}
    del expected['env']['skillInitCurriculum']
    assert existing == expected


def test_config_is_distance_current_saturation_without_warmup_or_bonus():
    base, new = configs()
    expected = copy.deepcopy(base)
    expected['env']['relationReward']['progress'] = {'kind': 'distance', 'delta': .5, 'sigma': 1.}
    expected['env']['relationReward']['subgoal_success_bonus'] = 0.
    expected['env']['relationReward']['success'].pop('z_tolerance')
    expected['env']['relationReward']['success'].pop('saturate_edge_rewards')
    expected['env']['relationReward']['success'].update(
        require_achieved_target_prerequisites=True,
        require_current_target_prerequisites=False,
        saturate_edge_rewards_while_current=True,
        current_saturation_z_tolerance=.001,
        current_success_reward=.2)
    del expected['env']['skillInitCurriculum']
    assert new == expected
    assert 'skillInitCurriculum' not in new['env']
    assert new['env']['skillInitProb'] == [0., .5, .1, .3, .1]
    cfg = new['env']['relationReward']
    assert cfg['subgoal_success_bonus'] == 0
    assert cfg['success']['saturate_edge_rewards_while_current'] is True
    assert cfg['success']['require_achieved_target_prerequisites'] is True
    assert cfg['success']['current_saturation_z_tolerance'] == .001
    assert cfg['success']['current_success_reward'] == .2
    validate_relation_config(cfg)
    check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(cfg)}, checkpoint_metadata(cfg))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['env']['relationReward'])}, checkpoint_metadata(cfg))


def test_numeric_examples_plateau_monotonic_tail_and_scale():
    distances = torch.tensor([0., .49, .5, 1., 1.5, 2., 3., 5., 1000.])
    source = torch.zeros(len(distances), 3)
    target = torch.stack([distances, torch.zeros_like(distances), torch.zeros_like(distances)], -1)
    p = reward.distance_progress(source, target)
    expected = torch.tensor([1., 1., 1., 2/3, .5, .4, 2/7, 2/11, 1/1000.5])
    torch.testing.assert_close(p, expected)
    assert (p[3:] < p[2:-1]).all()
    assert (p > 0).all() and (p <= 1).all()
    torch.testing.assert_close(reward.distance_progress(source, target, delta=0, sigma=2), 1/(1+distances/2))


def test_position_only_ignores_motion_dt_and_z_for_arbitrary_edges():
    # Rows can represent H->O, O->G, or O_b->O_a: no type input exists.
    source = torch.tensor([[0., 0., 100.], [1., 2., -20.], [-3., -2., 17.]])
    target = source + torch.tensor([1., 0., 1000.])
    cfg = {'kind': 'distance', 'delta': .5, 'sigma': 1.}
    a = reward.relation_progress(None, source, target, 0., cfg)
    b = reward.relation_progress(source + 1e6, source, target, -99., cfg)
    torch.testing.assert_close(a, torch.full((3,), 2/3))
    torch.testing.assert_close(a, b)
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(a, reward.distance_progress(source @ rot + 3, target @ rot + 3))
    torch.testing.assert_close(a, reward.distance_progress(target, source))


def test_runtime_keeps_prerequisites_state_and_history_but_never_pins(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Distance progress must never call motion or pinning functions')
    for name in ('direction_progress', 'velocity_progress', 'pin_progress', 'approach_satisfaction'):
        monkeypatch.setattr(reward, name, forbidden)
    cfg = configs()[1]['env']['relationReward']
    graph = compile_carry_subgoal(2, 3)
    runtime = reward.RelationRuntime(1, graph, cfg, 'cpu')
    before = torch.tensor([[1., 0., 0., 0.]])
    runtime.reset(torch.tensor([0]), before, at_z_error=torch.zeros(1, 2))
    phi = torch.tensor([[1., 1., 1., 1.]])
    p = torch.tensor([[.1, .2, .3, .4]])
    result = runtime.step(phi, p, at_z_error=torch.tensor([[0., .002]]))
    g = reward.relation_gate(before)
    activation = torch.tensor([[1., g[0, 0], 1., g[0, 2]]])
    torch.testing.assert_close(result['activation'], activation)
    torch.testing.assert_close(result['pinned_progress'], p)
    assert not result['progress_blend'].any()
    torch.testing.assert_close(result['raw_state_component'], .2 * activation * phi)
    torch.testing.assert_close(result['raw_progress_component'], .2 * activation * p)
    saturated = result['saturation_active'][:, graph.edge_owner]
    torch.testing.assert_close(result['state_component'], torch.where(saturated, .2, .2 * activation * phi))
    torch.testing.assert_close(result['progress_component'], torch.where(saturated, .2, .2 * activation * p))
    torch.testing.assert_close(result['success_bonus'], torch.tensor([[.2, 0.]]))
    assert result['first_success_bonus'].tolist() == [[0., 0.]]
    torch.testing.assert_close(result['current_success_reward'], torch.tensor([[.2, 0.]]))
    assert result['current_success_state'].tolist() == [[True, False]]
    # Current saturation must disappear immediately after At/Z success is lost.
    again = runtime.step(torch.zeros_like(phi), torch.full_like(p, .1), at_z_error=torch.ones(1, 2))
    assert not again['success_bonus'].any()
    assert not again['saturation_active'].any()
    assert again['agent_task_reward'][0, 0].item() < .8
    assert again['edge_reward'][0, 2:].max() <= .020001
    assert not runtime.phi.any()


@pytest.mark.parametrize('at,z,success', [
    (.9, .001, True), (.9, -.001, True), (.8999, 0., False),
    (1., .00101, False), (1., -.00101, False)])
def test_current_success_is_at_and_z_only(at, z, success):
    cfg = configs()[1]['env']['relationReward']
    runtime = reward.RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    runtime.reset(torch.tensor([0]), torch.zeros(1, 2), at_z_error=torch.zeros(1, 1))
    result = runtime.step(torch.tensor([[0., at]]), torch.zeros(1, 2),
                          at_z_error=torch.tensor([[z]]))
    assert result['current_success_state'].item() == success
    assert result['saturation_active'].item() == success
    if success:
        assert result['agent_task_reward'].item() == pytest.approx(1.)
    else:
        # The ordinary soft prerequisite sigmoid is nonzero but negligible.
        assert abs(result['agent_task_reward'].item()) < 1e-10
    # Holding is absent, so saturation cannot fabricate relation history.
    assert not runtime.achieved.any()
    assert not runtime.done.any()
    suffix = runtime.suffix()[:, :-1].reshape(1, 2, 4)
    torch.testing.assert_close(suffix[..., 0], torch.tensor([[0., at]]))
    assert suffix[0, 0, 2].item() == 0


def test_current_saturation_does_not_follow_latched_done():
    cfg = configs()[1]['env']['relationReward']
    runtime = reward.RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    runtime.reset(torch.tensor([0]), torch.tensor([[1., 0.]]), at_z_error=torch.zeros(1, 1))
    success = runtime.step(torch.tensor([[0., 1.]]), torch.zeros(1, 2),
                           at_z_error=torch.zeros(1, 1))
    assert runtime.done.item() and success['saturation_active'].item()
    released = runtime.step(torch.zeros(1, 2), torch.zeros(1, 2),
                            at_z_error=torch.zeros(1, 1))
    assert runtime.done.item()  # diagnostic/observation history is still latched
    assert not released['saturation_active'].item()
    assert not released['edge_reward'].any()


@pytest.mark.parametrize('update', [
    {'saturate_edge_rewards': True},
    {'saturate_edge_rewards_while_current': 1},
    {'current_saturation_z_tolerance': 0},
    {'current_saturation_z_tolerance': None},
    {'current_success_reward': -1},
    {'current_success_reward': True},
])
def test_invalid_current_saturation_config_rejected(update):
    cfg = configs()[1]['env']['relationReward']
    cfg['success'].update(update)
    with pytest.raises(ValueError):
        validate_relation_config(cfg)


@pytest.mark.parametrize('bonus', [-1, True, float('nan'), float('inf')])
def test_success_bonus_may_be_zero_but_not_invalid(bonus):
    cfg = configs()[1]['env']['relationReward']
    cfg['subgoal_success_bonus'] = bonus
    with pytest.raises(ValueError, match='subgoal_success_bonus'):
        validate_relation_config(cfg)


@pytest.mark.parametrize('key,value', [
    ('delta', -1), ('delta', True), ('delta', float('nan')), ('delta', '.5'),
    ('sigma', 0), ('sigma', -1), ('sigma', True), ('sigma', float('inf')), ('sigma', None)])
def test_invalid_distance_parameters(key, value):
    cfg = configs()[1]['env']['relationReward']
    cfg['progress'][key] = value
    with pytest.raises(ValueError, match=key):
        validate_relation_config(cfg)
    with pytest.raises(ValueError, match=key):
        reward.distance_progress(torch.zeros(1, 3), torch.ones(1, 3), **{key: value})


@pytest.mark.parametrize('extra', ['normalization_epsilon', 'target_speed', 'velocity_scale',
                                  'approach_radius', 'at_approach_radius'])
def test_distance_rejects_obsolete_progress_options(extra):
    cfg = configs()[1]['env']['relationReward']
    cfg['progress'][extra] = .5
    with pytest.raises(ValueError, match='distance progress accepts only'):
        validate_relation_config(cfg)


@pytest.mark.parametrize('kind', ['velocity', 'direction'])
def test_distance_parameters_do_not_silently_affect_other_modes(kind):
    with pytest.raises(ValueError, match='delta/sigma'):
        validate_relation_config({'mode': 'state_relation_v0', 'progress': {'kind': kind, 'delta': .5}})


class ToyCarry(CarryRelationMixin):
    def _assigned_box_values(self, values):
        return values

    def _evaluate_relations(self):
        return torch.tensor([[.3, .1, 1., .2]]), {
            'goal_xy_error': torch.tensor([[1., .25]]), 'goal_z_error': torch.ones(1, 2)}

    def _record_relation_diagnostics(self, diag, result, previous_satisfied, live, objects, ph, pa):
        self.raw_progress = torch.stack([ph, pa], -1).flatten(1)
        self.result = result


def test_task_uses_root_box_and_box_target_current_xy_without_motion():
    task = ToyCarry()
    task.num_envs, task.num_agents, task.dt = 1, 2, .1
    task._relation_cfg = configs()[1]['env']['relationReward']
    task.relation_runtime = reward.RelationRuntime(1, compile_carry_subgoal(2, 3), task._relation_cfg, 'cpu')
    before = torch.tensor([[.8, .1, .99, .2]])
    task.relation_runtime.reset(torch.tensor([0]), before, at_z_error=torch.ones(1, 2))
    task._humanoid_root_states = torch.tensor([[[0., 0., 100.], [3., 0., -100.]]])
    task._box_states = torch.tensor([[[.5, 0., .2], [5., 0., .3]]])
    task._tar_pos = torch.tensor([[[1.5, 0., 20.], [5.25, 0., 30.]]])
    task._prev_root_pos = task._humanoid_root_states.clone() + 1000
    task._prev_box_pos = task._box_states.clone() - 1000
    task._power_reward = task._agent_collision_penalty = task._box_vel_penalty = False
    task.rew_buf, task.extras = torch.zeros(2), {}
    task._reward_term_sums, task._reward_term_count = torch.zeros(9), 0
    task._compute_relation_reward(None)
    expected = torch.tensor([[1., 2/3, .4, 1.]])
    torch.testing.assert_close(task.raw_progress, expected)
    torch.testing.assert_close(task.result['pinned_progress'], expected)
    torch.testing.assert_close(task.result['progress_component'], .2 * task.result['activation'] * expected)
