"""Direction-only ablation: unchanged state/gates/scales and old configs."""
import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.relation_reward import (
    RelationRuntime, direction_progress, relation_progress, velocity_progress,
    relation_gate)
from env.tasks.multi_agent.relation_task import CarryRelationMixin
from utils.relation_task_spec import (
    STATE_MODE, compile_carry_subgoal, validate_relation_config,
    checkpoint_metadata, check_checkpoint_metadata)


@pytest.mark.parametrize('velocity,expected', [
    ([1.5, 0., 0.], 1.), ([.5, 0., 0.], 1.), ([.1, 0., 0.], 1.),
    ([.0001, 0., 0.], 1.), ([1., 1., 0.], 1 / math.sqrt(2)),
    ([0., 1., 0.], 0.), ([-1., 0., 0.], 0.), ([0., 0., 0.], 0.),
    ([0., 0., 1.], 0.), ([.1, 0., 10.], 1.),
])
def test_direction_speed_angle_and_xy_invariance(velocity, expected):
    prev = torch.zeros(1, 3)
    cur = torch.tensor([velocity]) * .1
    goal = cur + torch.tensor([[2., 0., 4.]])
    result = direction_progress(prev, cur, goal, .1)
    torch.testing.assert_close(result, torch.tensor([expected]))
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(result, direction_progress(prev @ rot, cur @ rot, goal @ rot, .1))


def test_no_distance_pinning_height_mask_or_nonfinite_zero_distance():
    cur = torch.zeros(2, 3)
    prev = cur - torch.tensor([[.01, 0., 0.], [0., 0., 0.]])
    # Low box at 10 cm: forward motion gets 1, rest still gets 0.
    goal = cur + torch.tensor([[.1, 0., 0.]])
    torch.testing.assert_close(direction_progress(prev, cur, goal, .1), torch.tensor([1., 0.]))
    torch.testing.assert_close(direction_progress(prev, cur, cur, .1), torch.zeros(2))
    for dt, eps in ((0., 1e-6), (.1, 0.)):
        with pytest.raises(ValueError):
            direction_progress(prev, cur, goal, dt, eps)


def test_dispatch_preserves_default_gaussian():
    prev = torch.zeros(1, 3)
    cur = torch.tensor([[.05, 0., 0.]])
    goal = torch.tensor([[2., 0., 0.]])
    torch.testing.assert_close(relation_progress(prev, cur, goal, .1),
                               velocity_progress(prev, cur, goal, .1))
    assert relation_progress(prev, cur, goal, .1, {'kind': 'direction'}).item() == 1.
    with pytest.raises(ValueError):
        relation_progress(prev, cur, goal, .1, {'kind': 'unknown'})


def test_direction_config_is_progress_only_and_checkpoint_guarded():
    cfg_dir = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    old = yaml.safe_load((cfg_dir / 'amp_humanoid_ma_carry_relation_state02_near.yaml').read_text())
    new = yaml.safe_load((cfg_dir / 'amp_humanoid_ma_carry_relation_state02_near_dir.yaml').read_text())
    old_reward, new_reward = old['env']['relationReward'], new['env']['relationReward']
    validate_relation_config(old_reward)
    validate_relation_config(new_reward)
    assert new_reward['progress'] == {'kind': 'direction', 'normalization_epsilon': 1e-6}
    expected = copy.deepcopy(old)
    expected['env']['relationReward']['progress'] = new_reward['progress']
    assert expected == new
    for cfg in (old_reward, new_reward):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(cfg)}, checkpoint_metadata(cfg))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old_reward)},
                                  checkpoint_metadata(new_reward))
    for progress in ({'kind': 'unknown'}, {'kind': 'direction', 'target_speed': 1.5},
                     {'kind': 'direction', 'velocity_scale': 5.},
                     {'kind': 'direction', 'normalization_epsilon': 0.}):
        with pytest.raises(ValueError):
            validate_relation_config({'mode': STATE_MODE, 'progress': progress})


class ToyCarry(CarryRelationMixin):
    """Exercise the actual reward integration without importing the simulator."""
    def _assigned_box_values(self, values):
        return values

    def _evaluate_relations(self):
        return self.test_phi, {}

    def _record_relation_diagnostics(self, diag, result, previous_satisfied, live, objects, ph, pa):
        self.test_raw = torch.stack([ph, pa], -1)
        self.test_result = result


def test_both_edges_use_direction_with_existing_gates_weights_and_success():
    task = ToyCarry()
    task.num_envs = task.num_agents = 1
    task.dt = .1
    task._relation_cfg = {'progress': {'kind': 'direction'},
                          'state_reward_weight': .2, 'progress_reward_weight': .2}
    task.relation_runtime = RelationRuntime(1, compile_carry_subgoal(1, 1),
                                           task._relation_cfg, 'cpu')
    before = torch.tensor([[.8, .2]])
    task.relation_runtime.reset(torch.tensor([0]), before)
    task.test_phi = torch.tensor([[.9, .4]])
    task._humanoid_root_states = torch.tensor([[[.01, 0., 0.]]])
    task._prev_root_pos = torch.zeros(1, 1, 3)
    task._box_states = torch.tensor([[[1.01, 0., 0.]]])
    task._prev_box_pos = torch.tensor([[[1., 0., 0.]]])
    task._tar_pos = torch.tensor([[[2., 0., 0.]]])
    task._power_reward = task._agent_collision_penalty = task._box_vel_penalty = False
    task.rew_buf = torch.zeros(1)
    task.extras = {}
    task._reward_term_sums = torch.zeros(9)
    task._reward_term_count = 0
    task._compute_relation_reward(None)
    torch.testing.assert_close(task.test_raw, torch.ones(1, 1, 2))
    torch.testing.assert_close(task.test_result['progress_component'], torch.tensor([[.2, .1]]))
    torch.testing.assert_close(task.test_result['state_component'], torch.tensor([[.18, .04]]))
    torch.testing.assert_close(task.rew_buf, torch.tensor([.52]))
    assert task.test_result['success_bonus'].item() == 0.
    # At rest only the existing soft-pinning remains, with previous-state gates.
    task._prev_root_pos = task._humanoid_root_states.clone()
    task._prev_box_pos = task._box_states.clone()
    task._compute_relation_reward(None)
    torch.testing.assert_close(task.test_raw, torch.zeros(1, 1, 2))
    gate = relation_gate(task.test_phi)
    torch.testing.assert_close(task.test_result['progress_component'],
                               .2 * torch.stack([gate[:, 0], gate[:, 0] * gate[:, 1]], -1))
