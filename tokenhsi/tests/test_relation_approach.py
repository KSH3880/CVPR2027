"""At-only XY approach blending leaves XYZ state, prerequisites and success intact."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.relation_reward import (
    RelationRuntime, approach_satisfaction, evaluate_at, pin_progress, relation_gate)
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import (
    STATE_MODE, compile_carry_subgoal, validate_relation_config,
    checkpoint_metadata, check_checkpoint_metadata)


def load_configs():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    stem = 'amp_humanoid_ma_carry_relation_state02_near_dir_success10'
    return tuple(yaml.safe_load((directory / (stem + suffix + '.yaml')).read_text())
                 for suffix in ('', '_approach'))


def test_config_changes_only_at_blend_and_preserves_checkpoint_contract():
    base, new = load_configs()
    expected = copy.deepcopy(base)
    expected['env']['relationReward']['progress']['at_approach_radius'] = .5
    assert new == expected
    base_cfg, cfg = base['env']['relationReward'], new['env']['relationReward']
    validate_relation_config(base_cfg)
    validate_relation_config(cfg)
    check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(cfg)}, checkpoint_metadata(cfg))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base_cfg)}, checkpoint_metadata(cfg))


def test_distance_table_bounds_and_monotonicity():
    distance = torch.tensor([0., .05, .1, .2, .3, .5, 1., 2., 3.])
    approach = approach_satisfaction(distance, .5)
    torch.testing.assert_close(approach, .25 / (distance.square() + .25))
    assert approach[0] == 1. and approach[5] == .5
    assert (approach[:-1] > approach[1:]).all()
    raw = torch.linspace(0., 1., 101)[:, None]
    total = pin_progress(raw, approach[None, :])
    assert ((total >= 0.) & (total <= 1.)).all()
    torch.testing.assert_close(total[0], approach)
    torch.testing.assert_close(total[-1], torch.ones_like(approach))
    assert (total[1:] >= total[:-1]).all()
    assert (total[:, :-1] >= total[:, 1:]).all()


@pytest.mark.parametrize('radius', [0., -1., float('nan'), float('inf'), None, True, '.5'])
def test_bad_approach_config_rejected(radius):
    with pytest.raises(ValueError, match='radius'):
        validate_relation_config({'mode': STATE_MODE,
                                  'progress': {'kind': 'direction', 'at_approach_radius': radius}})


def test_approach_requires_direction_and_valid_radius():
    with pytest.raises(ValueError, match='requires direction'):
        validate_relation_config({'mode': STATE_MODE, 'progress': {'at_approach_radius': .5}})
    for radius in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='radius'):
            approach_satisfaction(torch.zeros(1), radius)


def test_only_at_progress_changes_not_state_history_or_holding():
    base, new = load_configs()
    graph = compile_carry_subgoal(2, 3)
    old = RelationRuntime(2, graph, base['env']['relationReward'], 'cpu')
    current = RelationRuntime(2, graph, new['env']['relationReward'], 'cpu')
    before = torch.tensor([[1., .9, .8, .1], [.95, 0., 1., 0.]])
    for runtime in (old, current):
        runtime.reset(torch.arange(2), before)
    phi = torch.tensor([[1., .4, .9, .3], [.7, .95, .95, .96]])
    raw = torch.tensor([[.1, 0., .5, .5], [1., .3, 0., 1.]])
    distance = torch.tensor([[0., .5], [.1, .05]])
    baseline = old.step(phi, raw)
    result = current.step(phi, raw, at_distance_xy=distance)
    for key in ('activation', 'state_component', 'success_bonus', 'first_success',
                'achieved_next', 'done_next', 'valid_next', 'gate_next', 'satisfied_next'):
        torch.testing.assert_close(result[key], baseline[key])
    torch.testing.assert_close(current.suffix(), old.suffix())
    torch.testing.assert_close(result['progress_component'][:, 0::2],
                               baseline['progress_component'][:, 0::2])
    a = approach_satisfaction(distance)
    expected = a + (1 - a) * raw[:, 1::2]
    torch.testing.assert_close(result['pinned_progress'][:, 1::2], expected)
    torch.testing.assert_close(result['progress_component'][:, 1::2],
                               .2 * relation_gate(before)[:, 0::2] * expected)
    again = current.step(phi, raw, at_distance_xy=distance)
    assert not again['success_bonus'].any()


def test_approach_credit_never_seeds_or_awards_xyz_success():
    cfg = load_configs()[1]['env']['relationReward']
    runtime = RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    phi = torch.tensor([[1., .4]])
    runtime.reset(torch.tensor([0]), phi)
    result = runtime.step(phi, torch.zeros_like(phi), at_distance_xy=torch.zeros(1, 1))
    assert result['pinned_progress'][0, 1] == 1.
    assert result['gate_next'][0, 1] < 1e-4
    assert not result['satisfied_next'][0, 1]
    assert not result['achieved_next'][0, 1]
    assert not result['done_next'].any()
    assert result['success_bonus'].item() == 0.


def test_missing_and_invalid_distances_fail_loudly():
    cfg = copy.deepcopy(load_configs()[1]['env']['relationReward'])
    cfg['diagnostics']['validate_tensors'] = True
    runtime = RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    phi = torch.zeros(1, 2)
    for distance in (None, torch.zeros(1), torch.zeros(1, 2),
                     torch.tensor([[-.1]]), torch.tensor([[float('nan')]]),
                     torch.zeros(1, 1, dtype=torch.float64)):
        with pytest.raises(ValueError, match='distances'):
            runtime.step(phi, phi, at_distance_xy=distance)


class ToyApproachCarry(CarryRelationMixin):
    def _assigned_box_values(self, values):
        return values

    def _evaluate_relations(self):
        at, _, _, xy, _ = evaluate_at(self._box_states, self._tar_pos, state_definition='box_near')
        phi = torch.stack([torch.ones_like(at), at], -1).flatten(1)
        return phi, {'goal_xy_error': xy}

    def _record_relation_diagnostics(self, diag, result, previous_satisfied, live, objects, ph, pa):
        self.test_raw = torch.stack([ph, pa], -1)
        self.test_result = result


def test_actual_mixin_uses_poststep_xy_distance_without_extra_at_gate():
    task = ToyApproachCarry()
    task.num_envs = task.num_agents = 2
    task.dt = .1
    task._relation_cfg = load_configs()[1]['env']['relationReward']
    task.relation_runtime = RelationRuntime(2, compile_carry_subgoal(2, 3), task._relation_cfg, 'cpu')
    before = torch.tensor([[.8, .9, .8, .9], [.8, .9, .8, .9]])
    task.relation_runtime.reset(torch.arange(2), before)
    task._humanoid_root_states = torch.zeros(2, 2, 3)
    task._prev_root_pos = task._humanoid_root_states.clone()
    task._box_states = torch.zeros(2, 2, 3)
    task._box_states[..., 2] = .3
    task._prev_box_pos = task._box_states.clone()  # stationary, so raw dir=0
    task._tar_pos = torch.zeros(2, 2, 3)
    distance = torch.tensor([[.1, .5], [0., 2.]])
    task._tar_pos[..., 0] = distance
    task._power_reward = task._agent_collision_penalty = task._box_vel_penalty = False
    task.rew_buf = torch.zeros(4)
    task.extras = {}
    task._reward_term_sums = torch.zeros(9)
    task._reward_term_count = 0
    task._compute_relation_reward(None)
    result = task.test_result
    assert not task.test_raw.any()
    # Previous At gate is ~0.95; it must NOT be reapplied to the approach blend.
    torch.testing.assert_close(result['pinned_progress'][:, 1::2], approach_satisfaction(distance))
    torch.testing.assert_close(result['progress_component'][:, 1::2], .1 * approach_satisfaction(distance))
    torch.testing.assert_close(result['progress_component'][:, 0::2], torch.full((2, 2), .1))
    torch.testing.assert_close(task.relation_runtime.phi[:, 1::2], torch.exp(-10 * (distance.square() + .09)))
    assert not result['first_success'].any()
    assert torch.isfinite(task.rew_buf).all()
