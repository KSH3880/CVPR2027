"""All-edge RSI progress, prerequisites and history for the retained experiments."""
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.relation_reward import (
    RelationRuntime, approach_satisfaction, evaluate_at, relation_gate)
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import (
    STATE_MODE, compile_carry_subgoal, validate_relation_config)


def load_config():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    return yaml.safe_load((directory / 'approach_rsi_all_edges.yaml').read_text())


def test_common_progress_formula_preserves_prerequisites_and_history():
    graph = compile_carry_subgoal(2, 3)
    current = RelationRuntime(2, graph, load_config()['env']['relationReward'], 'cpu')
    before = torch.tensor([[.99, .2, .3, .1], [.8, .1, .95, .2]])
    phi = torch.tensor([[.3, .95, .99, .95], [.99, .95, .7, .95]])
    raw = torch.tensor([[0., .2, .7, .4], [.5, .8, 1., 0.]])
    distances = torch.tensor([[.5, .1, 0., .5], [2., 0., .1, 3.]])
    current.reset(torch.arange(2), before)
    result = current.step(phi, raw, edge_distance_xy=distances)
    a = approach_satisfaction(distances, .5)
    expected_progress = a + (1 - a) * raw
    activation = torch.ones_like(phi)
    activation[:, 1::2] = relation_gate(before)[:, 0::2]
    torch.testing.assert_close(result['activation'], activation)
    torch.testing.assert_close(result['state_component'], .2 * activation * phi)
    torch.testing.assert_close(result['progress_blend'], a)
    torch.testing.assert_close(result['pinned_progress'], expected_progress)
    torch.testing.assert_close(result['progress_component'], .2 * activation * expected_progress)
    assert result['first_success'].tolist() == [[True, False], [False, True]]
    torch.testing.assert_close(result['success_bonus'], torch.tensor([[10., 0.], [0., 10.]]))
    assert result['activation'][0, 3] < 1e-5
    again = current.step(phi, raw, edge_distance_xy=distances)
    assert again['success_bonus'][0, 0] == 0  # repeat success does not re-award the bonus


@pytest.mark.parametrize('radius', [0., -1., float('nan'), float('inf'), None, True, '.5'])
def test_bad_common_radius_rejected(radius):
    with pytest.raises(ValueError, match='radius'):
        validate_relation_config({'mode': STATE_MODE,
            'progress': {'kind': 'direction', 'approach_radius': radius}})


def test_common_blend_requires_direction_and_rejects_at_only_option():
    with pytest.raises(ValueError, match='distance progress'):
        validate_relation_config({'mode': STATE_MODE, 'progress': {'approach_radius': .5}})
    with pytest.raises(ValueError, match='Unsupported relationReward.progress'):
        validate_relation_config({'mode': STATE_MODE, 'progress': {
            'kind': 'direction', 'approach_radius': .5, 'at_approach_radius': .5}})


def test_edge_distances_are_required_and_validated():
    cfg = load_config()['env']['relationReward']
    cfg['diagnostics']['validate_tensors'] = True
    runtime = RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    phi = torch.zeros(1, 2)
    for distances in (None, torch.zeros(1), torch.zeros(1, 1),
                      torch.tensor([[-1., 0.]]), torch.tensor([[0., float('nan')]]),
                      torch.zeros(1, 2, dtype=torch.float64)):
        with pytest.raises(ValueError, match='distances'):
            runtime.step(phi, phi, edge_distance_xy=distances)


class ToyCarry(CarryRelationMixin):
    def _assigned_box_values(self, values):
        return values

    def _evaluate_relations(self):
        at, _, _, xy, _ = evaluate_at(self._box_states, self._tar_pos, state_definition='box_near')
        return torch.stack([torch.ones_like(at), at], -1).flatten(1), {'goal_xy_error': xy}

    def _record_relation_diagnostics(self, diag, result, previous_satisfied, live, objects, ph, pa):
        self.test_raw = torch.stack([ph, pa], -1).flatten(1)
        self.test_result = result


def test_task_passes_poststep_root_box_and_box_target_xy_distances():
    task = ToyCarry()
    task.num_envs, task.num_agents, task.dt = 1, 2, .1
    task._relation_cfg = load_config()['env']['relationReward']
    task.relation_runtime = RelationRuntime(1, compile_carry_subgoal(2, 3), task._relation_cfg, 'cpu')
    before = torch.tensor([[.8, .1, .99, .2]])
    task.relation_runtime.reset(torch.tensor([0]), before)
    task._humanoid_root_states = torch.tensor([[[0., 0., 1.], [1., 0., 1.]]])
    task._prev_root_pos = task._humanoid_root_states.clone()
    task._prev_root_pos[..., 0] -= .1
    task._box_states = torch.tensor([[[.5, 0., .2], [1., 0., .3]]])
    task._prev_box_pos = task._box_states.clone()
    task._tar_pos = torch.tensor([[[1., 0., .2], [3., 0., .3]]])
    task._power_reward = task._agent_collision_penalty = task._box_vel_penalty = False
    task.rew_buf, task.extras = torch.zeros(2), {}
    task._reward_term_sums, task._reward_term_count = torch.zeros(9), 0
    task._compute_relation_reward(None)
    distances = torch.tensor([[.5, .5, 0., 2.]])
    a = approach_satisfaction(distances)
    torch.testing.assert_close(task.test_raw, torch.tensor([[1., 0., 0., 0.]]))
    torch.testing.assert_close(task.test_result['progress_blend'], a)
    expected_progress = a + (1 - a) * task.test_raw
    torch.testing.assert_close(task.test_result['pinned_progress'], expected_progress)
    phi = task.relation_runtime.phi
    activation = torch.ones_like(phi)
    activation[:, 1::2] = relation_gate(before)[:, 0::2]
    expected_reward = (.2 * activation * (phi + expected_progress)).reshape(1, 2, 2).sum(-1)
    torch.testing.assert_close(task.rew_buf, expected_reward.flatten())
