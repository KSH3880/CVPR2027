"""Opt-in current Holding/At/Z success and paid-only per-agent saturation."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.relation_reward import RelationRuntime, relation_gate, box_speed_penalty
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import (
    compile_carry_subgoal, validate_relation_config, checkpoint_metadata, check_checkpoint_metadata)


def configs():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    return [yaml.safe_load((directory / (name + '.yaml')).read_text()) for name in
            ('approach_rsi_all_edges', 'approach_rsi_all_edges_success_sat')]


def runtime(n=1, m=1):
    return RelationRuntime(n, compile_carry_subgoal(m, m), configs()[1]['env']['relationReward'], 'cpu')


def step(r, phi, z):
    return r.step(phi, torch.zeros_like(phi), edge_distance_xy=torch.ones_like(phi), at_z_error=z)


def test_config_changes_only_success_and_preserves_checkpoint_contract():
    base, new = configs()
    expected = copy.deepcopy(base)
    expected['env']['relationReward']['success'].update(
        require_achieved_target_prerequisites=False,
        require_current_target_prerequisites=True, z_tolerance=.001, saturate_edge_rewards=True)
    assert new == expected
    cfg = new['env']['relationReward']
    validate_relation_config(cfg)
    metadata = checkpoint_metadata(cfg)
    check_checkpoint_metadata({'relation_metadata': metadata}, metadata)
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['env']['relationReward'])}, metadata)


@pytest.mark.parametrize('holding,at,z,success', [
    (.9, .9, .001, True), (.9, .9, -.001, True),
    (.899, 1., 0., False), (1., .899, 0., False),
    (1., 1., .00101, False), (1., 1., -.00101, False)])
def test_current_state_thresholds_without_previous_holding(holding, at, z, success):
    r = runtime()
    r.reset(torch.tensor([0]), torch.zeros(1, 2), at_z_error=torch.zeros(1, 1))
    result = step(r, torch.tensor([[holding, at]]), torch.tensor([[z]]))
    assert result['first_success'].item() == success
    assert result['success_bonus'].item() == (10 if success else 0)
    assert r.done.item() == success
    if success:
        assert result['agent_task_reward'].item() == pytest.approx(10.8)


def test_holding_history_alone_no_longer_suffices():
    r = runtime()
    r.reset(torch.tensor([0]), torch.tensor([[1., 0.]]), at_z_error=torch.zeros(1, 1))
    assert r.achieved[0, 0]
    result = step(r, torch.tensor([[0., 1.]]), torch.zeros(1, 1))
    assert not result['first_success'].any()
    assert not r.done.any()


def test_saturation_is_paid_only_latched_independent_and_once():
    r = runtime(2, 2)
    r.reset(torch.arange(2), torch.zeros(2, 4), at_z_error=torch.zeros(2, 2))
    phi = torch.tensor([[1., 1., .8, .8], [.8, .8, 1., 1.]])
    result = step(r, phi, torch.zeros(2, 2))
    assert result['success_bonus'].tolist() == [[10., 0.], [0., 10.]]
    released = torch.zeros_like(phi)
    step(r, released, torch.ones(2, 2))
    result = step(r, released, torch.ones(2, 2))
    done_edges = r.done[:, r.graph.edge_owner]
    torch.testing.assert_close(result['edge_reward'][done_edges], torch.full((4,), .4))
    torch.testing.assert_close(result['agent_task_reward'][r.done], torch.full((2,), .8))
    assert not result['success_bonus'].any()
    assert not result['valid_next'].any()
    assert not result['satisfied_next'].any()
    assert not r.phi.any()
    torch.testing.assert_close(result['gate_next'], relation_gate(released))
    assert (result['activation'][:, 1::2] < 1e-6).all()
    assert (result['raw_progress_component'][:, 1::2] < 1e-6).all()
    assert not result['raw_state_component'].any()
    assert (result['edge_reward'][~done_edges] < .4).all()
    suffix = r.suffix()[:, :-2].reshape(2, 4, 4)
    torch.testing.assert_close(suffix[..., 0], released)
    torch.testing.assert_close(suffix[..., 1], relation_gate(released))
    assert not suffix[..., 2].any()
    # Selective reset clears only the reset scene's latch and allows a new bonus.
    saved = r.suffix()[1].clone()
    r.reset(torch.tensor([0]), torch.zeros(1, 4), at_z_error=torch.zeros(1, 2))
    assert not r.done[0].any()
    torch.testing.assert_close(r.suffix()[1], saved)
    result = step(r, phi, torch.zeros(2, 2))
    assert result['success_bonus'].tolist() == [[10., 0.], [0., 0.]]


def test_initial_success_uses_strict_predicate_without_bonus():
    r = runtime(3)
    phi = torch.tensor([[1., 1.], [1., 1.], [.8, 1.]])
    z = torch.tensor([[0.], [.002], [0.]])
    r.reset(torch.arange(3), phi, at_z_error=z)
    assert r.done.flatten().tolist() == [True, False, False]
    result = step(r, phi, z)
    assert not result['success_bonus'].any()
    assert result['agent_task_reward'][0].item() == pytest.approx(.8)


def test_pre_success_rewards_match_base_exactly():
    base_cfg = configs()[0]['env']['relationReward']
    new = runtime()
    old = RelationRuntime(1, new.graph, base_cfg, 'cpu')
    phi = torch.tensor([[.7, .6]])
    for r in (old, new):
        r.reset(torch.tensor([0]), phi, at_z_error=torch.zeros(1, 1))
    a = step(old, phi, torch.zeros(1, 1))
    b = step(new, phi, torch.zeros(1, 1))
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    torch.testing.assert_close(old.suffix(), new.suffix())


@pytest.mark.parametrize('tolerance', [0, -1, True, '.001', None, float('nan'), float('inf')])
def test_invalid_z_tolerance_rejected(tolerance):
    cfg = configs()[1]['env']['relationReward']
    cfg['success']['z_tolerance'] = tolerance
    with pytest.raises(ValueError, match='z_tolerance'):
        validate_relation_config(cfg)


def test_inconsistent_success_flags_rejected():
    for update in ({'require_achieved_target_prerequisites': True},
                   {'require_current_target_prerequisites': False},
                   {'saturate_edge_rewards': 1}):
        cfg = configs()[1]['env']['relationReward']
        cfg['success'].update(update)
        with pytest.raises(ValueError):
            validate_relation_config(cfg)


def test_missing_z_data_fails_in_reset_and_step():
    r = runtime()
    with pytest.raises(ValueError, match='Z errors'):
        r.reset(torch.tensor([0]), torch.ones(1, 2))
    with pytest.raises(ValueError, match='Z errors'):
        step(r, torch.ones(1, 2), None)


class ToyCarry(CarryRelationMixin):
    def _assigned_box_values(self, values):
        return values

    def _evaluate_relations(self):
        return self.test_phi, {'goal_xy_error': torch.zeros(1, 2), 'goal_z_error': self.test_z}

    def _record_relation_diagnostics(self, *args):
        self.result = args[1]


def test_task_wires_z_and_keeps_all_external_penalties():
    task = ToyCarry()
    task.num_envs, task.num_agents, task.dt = 1, 2, .1
    task.relation_runtime = runtime(1, 2)
    task._relation_cfg = task.relation_runtime.config
    task.test_phi = torch.ones(1, 4)
    task.test_z = torch.tensor([[0., .002]])
    task._box_states = torch.tensor([[[.3, 0., .2], [.3, 0., .2]]])
    task._prev_box_pos = task._box_states.clone()
    task._prev_box_pos[..., 0] = 0.
    task._tar_pos = task._box_states.clone()
    task._humanoid_root_states = task._box_states.clone()
    task._prev_root_pos = task._humanoid_root_states.clone()
    task._power_reward = task._agent_collision_penalty = task._box_vel_penalty = True
    task._power_coefficient, task._agent_collision_coeff, task._agent_collision_dist = .0005, .5, .7
    task._box_vel_pen_coeff, task._box_vel_pen_thre = 1., 2.5
    task.dof_force_tensor = torch.ones(1, 2, 2) * 2
    task._dof_vel = torch.ones(1, 2, 2) * 3
    task.rew_buf, task.extras = torch.zeros(2), {}
    task._reward_term_sums, task._reward_term_count = torch.zeros(9), 0
    task._compute_relation_reward(lambda roots, distance: torch.ones(1, 2))
    assert task.result['first_success'].tolist() == [[True, False]]
    penalty = box_speed_penalty(task._prev_box_pos, task._box_states, task.dt)
    expected = task.result['agent_task_reward'] - .006 - .5 + penalty
    torch.testing.assert_close(task.rew_buf, expected.flatten())
    terms = task.extras['reward_terms']
    torch.testing.assert_close(terms[:, :8].sum(-1), terms[:, 8])
