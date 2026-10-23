"""PUSH addition preserves the native Stage1 AT, scene and original HOLDING."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.stage1_push import validate_env, sample_boxes_and_goals, check_checkpoint, load_bank
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_context_spec import HOLDING, AT
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges, interaction_own_success
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.humanoid_ma_stage1_push import PushRuntime


def config():
    return yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push.yaml').read_text())['env']


def test_native_at_config_and_network_are_unchanged():
    env = config(); validate_env(env)
    native = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    assert env['relationReward'] == native['relationReward']
    root = Path('tokenhsi/data/cfg/train/rlg')
    old, new = [yaml.safe_load((root/f'{name}.yaml').read_text()) for name in ('stage1_v2_fixed', 'stage1_push')]
    assert old['params']['network'] == new['params']['network']
    assert len(load_bank(env)['candidates']) == 11


def test_four_real_boxes_straight_uniform_goals_and_separation():
    env = config(); origins = torch.randn(20000, 3); origins[:, 2] = 0
    torch.manual_seed(42); boxes, goals, forward, distance = sample_boxes_and_goals(len(origins), 'cpu', origins, env)
    assert boxes.shape == (20000, 4, 13)
    assert distance.min() >= 1. and distance.max() <= 5.
    assert abs(distance.mean().item()-3.) < .03
    for start in range(1, 5):
        assert abs(((distance >= start)&(distance < start+1)).float().mean().item()-.25) < .015
    torch.testing.assert_close(goals-boxes[:, :2, :3], forward*distance[..., None], atol=2e-6, rtol=0)
    assert (goals[..., 2] == .65).all()
    assert torch.cdist(boxes[..., :2], boxes[..., :2]).masked_fill(torch.eye(4).bool()[None], 100).min() > 12.


def test_contact_requires_actual_hit_but_at_matches_native_success():
    env = config(); graph = compose_canonical_graph(torch.full((2, 2), 3))
    task = SimpleNamespace(_push_hits=torch.zeros(2, 2, dtype=torch.bool))
    push = PushRuntime(task, 2, graph, env['relationReward'], 'cpu')
    native = Stage1ContextRuntime(2, graph, env['relationReward'], 'cpu')
    phi = torch.ones(2, 4); z = torch.zeros_like(phi)
    at = graph.edge_relation == AT; hold = graph.edge_relation == HOLDING
    phi[at] = torch.exp(torch.tensor(-10*.19**2))
    phi[1][at[1]] = torch.exp(torch.tensor(-10*.21**2))
    p = push.step(phi, phi, z, z, z); q = native.step(phi, phi, z, z, z)
    assert torch.equal(p['own_success'][at], q['own_success'][at])
    assert not p['own_success'][hold].any()  # near hands alone are not force contact
    assert push.done[0].all() and not push.done[1].any()
    assert p['reward_saturated'][0].all()  # native paired placement saturation
    task._push_hits[:] = True
    assert push.step(phi, phi, z, z, z)['own_success'][hold].all()
    z[0, at[0]] = .002
    assert not push.step(phi, phi, z, z, z)['own_success'][0, at[0]].any()


def test_checkpoint_cannot_cross_native_holding_and_push_or_goal_ranges():
    env = config(); weights = {'relation_experiment_config': {'env': deepcopy(env)}}
    check_checkpoint(weights, env)
    wrong = deepcopy(env); wrong['pushGoalRandomization']['distance_range'] = [2., 5.]
    with pytest.raises(ValueError): check_checkpoint(weights, wrong)
    native = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    with pytest.raises(ValueError): check_checkpoint(weights, native)
    with pytest.raises(ValueError): check_checkpoint({'relation_experiment_config':{'env': native}}, env)


def test_actual_contact_geometry_requires_both_hands_force_and_height():
    from utils.push_point import local_points, point_reward_contact
    env = config(); box = torch.zeros(1, 1, 13); box[..., 2] = .65; box[..., 6] = 1.
    side = torch.full((1, 1), 2); point = local_points(side, [1.3]*3, torch.full((1, 1), .6175))
    hands = torch.tensor([[[[-.2, -.65, .6175], [.2, -.65, .6175]]]])
    force = torch.zeros_like(hands); box_force = torch.tensor([[[0., 2., 0.]]])
    limits = dict(hand_distance=.18, min_hand_force=.5, min_box_force=.5)
    def contact(): return point_reward_contact(hands, force, box, box_force, [1.3]*3, limits, side, point, env['pushPointReward']).item()
    assert not contact()
    force[..., 1] = -1.
    assert contact()
    force[..., 0, 1] = 0.
    assert not contact()
    force[..., 0, 1] = -1.; hands[..., 2] += .04
    assert not contact()


def test_no_force_config_only_changes_force_gate_and_checkpoint_contract():
    old = config()
    env = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml').read_text())['env']
    validate_env(env)
    expected = deepcopy(old); spec = expected['pushTask']
    spec.update(revision=2, require_contact_force=False)
    spec.pop('min_hand_force_n'); spec.pop('min_box_force_n')
    assert env == expected
    weights = {'relation_experiment_config': {'env': deepcopy(old)}}
    with pytest.raises(ValueError): check_checkpoint(weights, env)
    weights['relation_experiment_config']['env'] = deepcopy(env)
    check_checkpoint(weights, env)
    with pytest.raises(ValueError): check_checkpoint(weights, old)


@pytest.mark.parametrize('force_value', [0., -100., 100.])
def test_no_force_contact_accepts_any_force_but_preserves_all_geometry(force_value):
    from utils.push_point import local_points, point_reward_contact
    env = config(); box = torch.zeros(1, 1, 13); box[..., 2] = .65; box[..., 6] = 1.
    side = torch.full((1, 1), 2); point = local_points(side, [1.3]*3, torch.full((1, 1), .6175))
    valid = torch.tensor([[[[-.2, -.65, .6175], [.2, -.65, .6175]]]])
    forces = torch.full_like(valid, force_value); box_force = torch.full((1, 1, 3), -force_value)
    limits = dict(hand_distance=.18, require_contact_force=False)
    def contact(hands): return point_reward_contact(hands, forces, box, box_force, [1.3]*3, limits, side, point, env['pushPointReward']).item()
    assert contact(valid)
    for z in (.585, .650):
        hands = valid.clone(); hands[..., 2] = z; assert contact(hands)
    hands = valid.clone(); hands[..., 2] = .66; assert not contact(hands)
    hands = valid.clone(); hands[..., 0] += .19; assert not contact(hands)
    hands = valid.clone(); hands[..., 0, 1] -= .25; hands[..., 1, 1] += .25
    assert not contact(hands)  # midpoint stays correct, each hand is too far from the face
