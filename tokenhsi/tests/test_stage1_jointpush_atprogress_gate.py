from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime, gate_push_at_progress
from utils.stage1_jointpush import compose_graph, contract, validate_env

ROOT = Path(__file__).resolve().parents[2]
BASE = 'stage1_jointpush_density3_cone90_finetune'
NAME = BASE + '_atprogress_gate'
ENV = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{NAME}.yaml').read_text())['env']


def test_config_changes_only_gate_and_run_name():
    baseline = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{BASE}.yaml').read_text())['env']
    validate_env(ENV)
    restored = deepcopy(ENV)
    assert restored['jointPush'].pop('at_progress_min_delta') == .001
    assert restored == baseline
    train = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/{NAME}.yaml').read_text())
    old = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/{BASE}.yaml').read_text())
    assert train['params']['config'].pop('name') == 'Stage1JointPushDensity3Cone90FinetuneATProgressGate'
    old['params']['config'].pop('name')
    assert train == old
    assert contract(ENV) != contract(baseline)


@pytest.mark.parametrize('threshold', [0., -.001, float('nan'), float('inf')])
def test_invalid_threshold(threshold):
    config = deepcopy(ENV)
    config['jointPush']['at_progress_min_delta'] = threshold
    with pytest.raises(ValueError, match='finite positive'):
        validate_env(config)


def test_strict_threshold_only_gates_push_at_with_shuffled_edges():
    graph = compose_graph(torch.tensor([[5, 3]]).repeat(5, 1), shuffle=True)
    raw = torch.full((5, 4), .4, dtype=torch.float64)
    previous = torch.zeros(5, 2, 3, dtype=torch.float64)
    previous[:, 0, 0] = torch.tensor([0., .0009, .001, .0011, -.002], dtype=torch.float64)
    goals = torch.zeros_like(previous)
    distance = torch.zeros_like(raw)
    distance[-1] = .004
    gated = gate_push_at_progress(raw, distance, previous, goals, graph, .001)
    mask = (graph.edge_relation == 7) & (graph.edge_owner == 0)
    torch.testing.assert_close(gated[mask], torch.tensor([0., 0., 0., .4, 0.], dtype=torch.float64))
    torch.testing.assert_close(gated[~mask], raw[~mask])


def test_horizontal_distance_reduction_not_displacement_or_height():
    graph = compose_graph(torch.full((3, 2), 5), shuffle=True)
    previous = torch.zeros(3, 2, 3)
    previous[..., 0] = 1.
    previous[..., 2] = 30.
    goals = torch.zeros_like(previous)
    distances = torch.tensor([1., 1.002, .998])[:, None].expand(3, 4)
    raw = torch.full((3, 4), .1)
    gated = gate_push_at_progress(raw, distances, previous, goals, graph, .001)
    mask = graph.edge_relation == 7
    torch.testing.assert_close(gated[mask].reshape(3, 2), torch.tensor([[0., 0.], [0., 0.], [.1, .1]]))
    torch.testing.assert_close(gated[~mask], raw[~mask])


def test_success_overrides_gate_and_other_reward_components_unchanged():
    graph = compose_graph(torch.full((2, 2), 5), shuffle=True)
    runtime = Stage1PushRuntime(2, graph, ENV['relationReward'], 'cpu', contact_weights=[.2]*3)
    phi = torch.full((2, 4), .1)
    phi[1] = torch.where(graph.edge_relation[1] == 7, 1., .1)
    raw = torch.full_like(phi, .4)
    zero = torch.zeros_like(phi)
    positions = torch.zeros(2, 2, 3)
    gated = gate_push_at_progress(raw, zero, positions, positions, graph, .001)
    runtime.contact.fill_(True)
    before = runtime.step(phi, raw, zero, zero, zero)
    after = runtime.step(phi, gated, zero, zero, zero)
    at = graph.edge_relation == 7
    torch.testing.assert_close(after['progress_component'][0][at[0]], torch.zeros(2))
    torch.testing.assert_close(after['agent_task_reward'][1], torch.full((2,), 1.2))
    for key in ['state_component', 'success_component', 'own_success', 'reward_saturated']:
        torch.testing.assert_close(after[key], before[key])
    torch.testing.assert_close(after['progress_component'][~at], before['progress_component'][~at])


def test_reset_history_has_no_cross_episode_displacement():
    graph = compose_graph(torch.full((2, 2), 5))
    goals = torch.zeros(2, 2, 3)
    previous = torch.zeros_like(goals)
    previous[..., 0] = 5.
    reset_boxes = previous.clone()
    reset_boxes[0, :, 0] = 2.
    previous[0] = reset_boxes[0]
    distance = (reset_boxes[..., :2] - goals[..., :2]).norm(dim=-1).gather(1, graph.edge_owner)
    raw = 1/(1+(distance-.1).clamp_min(0))
    gated = gate_push_at_progress(raw, distance, previous, goals, graph, .001)
    assert not gated[graph.edge_relation == 7].any()
