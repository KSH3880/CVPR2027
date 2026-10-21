from copy import deepcopy
from pathlib import Path

import isaacgym
import pytest
import torch
import yaml

from env.tasks.multi_agent.bridge_geometry import compute_agent_cpa_collision_penalty
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.humanoid_ma_bridge import HumanoidMABridge
from utils.edge_scenario_spec import sample_graph


ENV = yaml.safe_load((Path(__file__).resolve().parents[1]
    / 'data/cfg/multi_agent/approach_stage2_fan_bridge.yaml').read_text())['env']


@pytest.mark.parametrize('velocity', [(-1., 1.), (0., 0.), (1., 1.)])
def test_close_agents_separating_stationary_or_matching_speed_have_no_penalty(velocity):
    roots = torch.tensor([[[0., 0., 2.4], [.5, 0., 2.4]]])
    speeds = torch.zeros_like(roots)
    speeds[0, :, 0] = torch.tensor(velocity)
    risk = compute_agent_cpa_collision_penalty(roots, speeds, .7, 1 / 30, .99)
    torch.testing.assert_close(risk, torch.zeros(1, 2), rtol=0, atol=0)


def test_head_on_risk_is_discounted_before_agents_enter_safe_distance():
    roots = torch.tensor([[[0., 0., 2.4], [4., 0., 2.4]]])
    speeds = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]])
    risk = compute_agent_cpa_collision_penalty(roots, speeds, .7, 1 / 30, .99)
    torch.testing.assert_close(risk, torch.full((1, 2), .99 ** 60))
    shifted = compute_agent_cpa_collision_penalty(roots + 20, speeds + 3, .7, 1 / 30, .99)
    torch.testing.assert_close(shifted, risk)


def test_glancing_risk_is_continuous_and_safe_miss_is_zero():
    roots = torch.tensor([[[0., 0., 2.4], [2., .35, 2.4]],
                          [[0., 0., 2.4], [2., .8, 2.4]]])
    speeds = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]]).expand(2, -1, -1)
    risk = compute_agent_cpa_collision_penalty(roots, speeds, .7, 1., 1.)
    expected = 2 / (4 + .35 ** 2) ** .5 * .5
    torch.testing.assert_close(risk[0], torch.full((2,), expected))
    torch.testing.assert_close(risk[1], torch.zeros(2), rtol=0, atol=0)


def test_self_pair_is_excluded():
    risk = compute_agent_cpa_collision_penalty(torch.zeros(2, 1, 3), torch.ones(2, 1, 3), .7, 1 / 30, .99)
    torch.testing.assert_close(risk, torch.zeros(2, 1), rtol=0, atol=0)


def test_bridge_relation_reward_replaces_distance_penalty_and_logs_single_collision():
    task = HumanoidMABridge.__new__(HumanoidMABridge)
    task.num_envs, task.num_agents = 2, 2
    task._relation_cfg = deepcopy(ENV['relationReward'])
    task._relation_cfg['diagnostics']['enabled'] = False
    task._edge_context = task._edge_interaction = task._edge_stage1 = True
    task._edge_ontop = True
    graph = sample_graph(task.num_envs, ENV['relationGraph'])
    task.relation_runtime = Stage1ContextRuntime(task.num_envs, graph, task._relation_cfg, 'cpu')
    phi = torch.full_like(task.relation_runtime.phi, .3)
    diag = {key: torch.zeros_like(phi) for key in ('progress', 'z_error', 'feet_height_error', 'region_error')}
    task._evaluate_relations = lambda: (phi, diag)
    task._humanoid_root_states = torch.zeros(2, 2, 13)
    task._humanoid_root_states[:, 1, 0] = .5
    task._humanoid_root_states[:, :, 7] = torch.tensor([[-1., 1.], [1., -1.]])
    task._box_states = torch.zeros(2, 3, 13)
    task._assigned_box_values = lambda values: values[:, :2]
    task._power_reward = task._box_vel_penalty = False
    task._agent_collision_coeff = ENV['agentCollisionCoeff']
    task._agent_collision_dist = ENV['agentCollisionDist']
    task._agent_collision_ttc_discount = ENV['agentCollisionTTCDiscount']
    task.dt = 1 / 30
    task._edge_steps = 0
    task._edge_ever_goal = torch.zeros(2, 2, dtype=torch.bool)
    task._edge_ever_scene = torch.zeros(2, dtype=torch.bool)
    task.REWARD_TERM_NAMES = ('edge_state', 'edge_progress', 'edge_success', 'power', 'collision', 'box_speed', 'total')
    task._reward_term_sums = torch.zeros(7)
    task._reward_term_count = 0
    task.rew_buf = torch.zeros(4)
    task.extras = {}
    task._agent_collision_penalty = False
    task._compute_reward(None)
    baseline = task.extras['reward_terms'].clone()
    task._agent_collision_penalty = True
    task._compute_reward(None)
    terms = task.extras['reward_terms']
    expected = torch.tensor([0., 0., -.5 * .99 ** 7.5, -.5 * .99 ** 7.5])
    torch.testing.assert_close(terms[:, 4], expected)
    torch.testing.assert_close(terms[:, -1] - baseline[:, -1], expected)
    torch.testing.assert_close(task.rew_buf, terms[:, -1])
    torch.testing.assert_close(terms[:, :4], baseline[:, :4], rtol=0, atol=0)
    assert task.REWARD_TERM_NAMES.count('collision') == 1
