from pathlib import Path
import torch
from torch import nn
import yaml

from utils.stage1_jointpush import sample_assets, compose_graph, box_profiles, validate_env
from utils.jointpush_scene import sample_layout, paths_clear, segment_distance
from utils.task_role_spec import task_graph_packet
from utils.edge_scenario_spec import classify_templates
from utils.jointpush_geometry import push_rsi_hand_targets
from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime, contact_state
from learning.multi_agent.stage1_push_network import Stage1PushBuilder

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage1_jointpush.yaml').read_text())


def test_sizes_and_task_mixture():
    validate_env(ENV)
    torch.manual_seed(42)
    sizes, weights = sample_assets(50000, ENV, 'cpu')
    kinds = torch.multinomial(weights.flatten(0, 1), 1).reshape(-1, 2)
    observed = torch.bincount(kinds.flatten(), minlength=6).float()/kinds.numel()
    torch.testing.assert_close(observed, torch.tensor([0., .1, .2, .25, .2, .25]), atol=.006, rtol=0)
    push = weights[..., 5].bool()
    dims = sizes[:, :2][push]
    assert len(box_profiles()) == 167
    assert (dims[:, :2].amin(-1)+1e-6 >= 1.1*dims[:, 2]).all()
    assert (sizes[:, 2:, :2] <= .8).all()
    assert (weights[..., 5][push] == 1).all()


def test_graph_task_packet_and_amp_identity():
    kinds = torch.tensor([[1, 5], [5, 3], [4, 2]])
    graph = compose_graph(kinds, shuffle=True)
    packet = task_graph_packet(graph, 3).reshape(3, 2, 5)
    torch.testing.assert_close(packet[..., 1].long(), torch.tensor([[0, 4], [4, 2], [3, 1]]))
    env = torch.arange(3).repeat_interleave(2)
    owner = torch.arange(2).repeat(3)
    torch.testing.assert_close(classify_templates(graph, env, owner), kinds.flatten())


def test_radial_layout_distances_and_full_goal_angles():
    torch.manual_seed(6)
    sizes, weights = sample_assets(1024, ENV, 'cpu', 'push')
    roots = torch.zeros(1024, 2, 3)
    roots[:, 0, 0], roots[:, 1, 0] = 2.5, -2.5
    boxes = torch.zeros(1024, 4, 13)
    boxes[..., 6] = 1
    push = weights[..., 5].bool()
    boxes, goals = sample_layout(roots, boxes, sizes, push, ENV['jointPush']['scene'])
    for delta in (boxes[:, :2, :2]-roots[..., :2], goals[..., :2]-boxes[:, :2, :2]):
        radius = delta.norm(dim=-1)
        assert ((radius >= 2.-1e-5) & (radius <= 10.+1e-5)).all()
    q = boxes[:, :2, 3:7]
    forward = torch.stack((-2*q[..., 2]*q[..., 3], 1-2*q[..., 2].square()), -1)
    delta = goals[..., :2]-boxes[:, :2, :2]
    cosine = (forward*delta).sum(-1)/delta.norm(dim=-1)
    assert (cosine < -.9).any() and (cosine > .9).any()
    assert (goals[..., :2].abs() > 4.5).any()
    torch.testing.assert_close(goals[..., 2], sizes[:, :2, 2]/2)
    assert paths_clear(boxes, sizes, goals, roots, push, ENV['jointPush']['scene']).all()


def test_crossing_paths_have_zero_gap():
    a, b = torch.tensor([[-1., 0.]]), torch.tensor([[1., 0.]])
    c, d = torch.tensor([[0., -1.]]), torch.tensor([[0., 1.]])
    assert segment_distance(a, b, c, d).item() == 0
    assert segment_distance(a, b, b+1, b+2).item() > 0


def test_contact_works_on_all_four_goal_sides():
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 2] = .75
    boxes[..., 6] = 1
    sizes = torch.tensor([2., 2., 1.5]).expand(1, 4, 3)
    goals = torch.tensor([[[4., 0., .75], [-4., 0., .75], [0., 4., .75], [0., -4., .75]]])
    spec = ENV['jointPush']['contact']
    hands = push_rsi_hand_targets(boxes, goals, sizes, dict(push=spec, contact_at=spec))
    normal = (goals-boxes[..., :3])/4
    phi, hit = contact_state(hands, boxes, goals, sizes, -normal[..., None, :].expand_as(hands), normal, spec)
    assert hit.all()
    torch.testing.assert_close(phi, torch.ones_like(phi))
    _, no_force = contact_state(hands, boxes, goals, sizes, torch.zeros_like(hands), normal, spec)
    assert not no_force.any()


def test_at_success_saturates_contact_without_contact_gate():
    graph = compose_graph(torch.full((1, 2), 5))
    runtime = Stage1PushRuntime(1, graph, ENV['relationReward'], 'cpu', contact_weights=[.2]*3)
    phi = torch.where(graph.edge_relation == 7, 1., 0.)
    zeros = torch.zeros_like(phi)
    result = runtime.step(phi, zeros, zeros, zeros, zeros)
    torch.testing.assert_close(result['agent_task_reward'], torch.full((1, 2), 1.2))
    assert runtime.done.all()
    assert not result['own_success'][graph.edge_relation == 12].any()


def test_user_checkpoint_loads_without_conversion():
    path = ROOT/'pth/Stage1PushFinetune_00010500.pth'
    if not path.exists():
        import pytest
        pytest.skip('User checkpoint is not distributed with the repository')
    builder = Stage1PushBuilder()
    builder.load(TRAIN['params']['network'])
    model = nn.Module()
    model.a2c_network = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1330,),
        value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=5., relation_reward_mode=ENV['relationReward']['mode'],
        relation_graph_spec=ENV['relationGraph'], device='cpu')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert len(checkpoint['model']) == 165
    assert checkpoint['amp_input_mean_std']['running_mean'].shape == (133,)
    assert TRAIN['params']['load_checkpoint'] is False
