from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from env.tasks.multi_agent.bridge_geometry import (
    bridge_control, bridge_schedule, bridge_trajectory, sample_bridge_layout, stage_height, terrain_fall,
    vertical_height)
from learning.multi_agent.bridge_network import BridgeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from env.tasks.multi_agent.push_reward import contact_geometry
from utils.bridge_spec import (
    SCENES, checkpoint_contract, predecessor_relations, push_episode_graph, sample_scenes, sample_templates,
    token_mapping, validate_env)
from utils.stage1_push import load_bank
from utils.edge_scenario_spec import compose_canonical_graph
from utils.stage2_spec import check_checkpoint
from utils.task_role_spec import task_graph_packet
from utils.unified_training import append_family, preserve_amp_labels, sample_family_matched

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_bridge_scripted.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage2_bridge_scripted.yaml').read_text())
EXPERIMENT = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_bridge_scripted.yaml').read_text())['experiment']
BRIDGE = ENV['bridge']
PUSH_ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_bridge_push.yaml').read_text())['env']


def test_config_contract_and_checkpoint_isolation():
    validate_env(ENV)
    ground = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix334_1200.yaml').read_text())['env']
    for key in ('relationReward', 'box', 'templateRsi', 'skillDiscProb', 'eval', 'plane'):
        assert ENV[key] == ground[key]
    ground['before'].pop('distance_mixture')
    assert ENV['before'] == ground['before']
    assert (ENV['numEnvs'], ENV['numAgents'], ENV['numObjects'], ENV['episodeLength']) == (2048, 2, 4, 600)
    assert TRAIN['params']['config']['minibatch_size'] == 4096 * 2
    check_checkpoint({'bridge_contract': checkpoint_contract(ENV)}, ENV)
    with pytest.raises(ValueError, match='Bridge checkpoint'):
        check_checkpoint({'before_contract': checkpoint_contract(ENV)}, ENV)
    for path, value in ((('bridge', 'mode'), 'push'), (('numObjects',), 5), (('envSpacing',), 10),
                        (('bridge', 'clearance'), 0.), (('box', 'reset', 'platformMode'), 'carry_payload_tokenhsi')):
        bad = deepcopy(ENV)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            validate_env(bad)


def test_templates_and_token_mapping():
    torch.manual_seed(0)
    templates = sample_templates(4096, 'random_scenario', BRIDGE['task_probabilities'], 'cpu')
    assert set(templates.unique().tolist()) == {1, 2}
    assert abs((templates == 1).float().mean() - .5) < .02
    assert (sample_templates(8, 'climb', BRIDGE['task_probabilities'], 'cpu') == 2).all()
    primary = torch.arange(2)[None] + templates * 2
    mapping = token_mapping(primary, 8, 8)
    assert torch.equal(mapping[:, :2], primary)
    assert ((mapping[:, 2:] == 8).sum(-1) == 1).all()
    distractor = torch.where(mapping[:, 2] == 8, mapping[:, 3], mapping[:, 2])
    assert ((distractor < 8) & (distractor != primary[:, 0]) & (distractor != primary[:, 1])).all()
    assert 0.45 < (mapping[:, 2] == 8).float().mean() < 0.55


def test_predecessor_relations():
    relations = predecessor_relations(torch.tensor([[-1., -1.], [1., -1.]]), 2)
    assert torch.equal(relations[0], torch.eye(2, dtype=torch.long))
    assert torch.equal(relations[1], torch.tensor([[1, 2], [0, 1]]))


def test_terrain_lookup_and_floor_start_exemption():
    origins = torch.zeros(1, 3)
    half_a = torch.tensor([[2., 3.]])
    xy = torch.tensor([[[0., 0.], [2.5, 0.], [2.5, 2.]]])
    near = stage_height(xy, origins, half_a, torch.tensor([1.5]))
    centers = torch.tensor([[[2.5, 0., .745]]])
    quat = torch.tensor([[[0., 0., 0., 1.]]])
    half = torch.tensor([[[.45, .6, .745]]])
    terrain = torch.maximum(near, vertical_height(xy, centers, quat, half, origins[:, 2]))
    torch.testing.assert_close(terrain, torch.tensor([[1.5, 1.49, 0.]]))

    bodies = 4
    contact = torch.zeros(1, 2, bodies, 3)
    body = torch.full((1, 2, bodies, 3), 1.)
    ground = torch.zeros(1, 2, bodies)
    feet = [2, 3]
    thresholds = torch.full((bodies,), .15)
    contact[..., 3, 2] = 50.
    body[..., 3, 2] = .05
    fallen = terrain_fall(contact, body, torch.tensor(feet), ground, thresholds, torch.tensor([5]),
                          origins[:, 2], torch.tensor([[False, True]]))
    assert fallen.tolist() == [[True, False]]
    contact[..., 0, 2] = 50.
    body[..., 0, 2] = .05
    fallen = terrain_fall(contact, body, torch.tensor(feet), ground, thresholds, torch.tensor([5]),
                          origins[:, 2], torch.tensor([[False, True]]))
    assert fallen.tolist() == [[True, True]]


def test_bridge_layout_invariants():
    torch.manual_seed(1)
    n, step = 4096, BRIDGE['stage_size_step']
    sizes_a = torch.arange(BRIDGE['stage_size_range'][0], BRIDGE['stage_size_range'][1] + .01, step)
    sizes_b = torch.arange(BRIDGE['far_size_range'][0], BRIDGE['far_size_range'][1] + .01, step)
    half_a = sizes_a[torch.randint(len(sizes_a), (n, 2))] / 2
    half_b = sizes_b[torch.randint(len(sizes_b), (n, 2))] / 2
    gap = torch.round((1.5 + torch.rand(n)) * 10) / 10
    bridge_half = torch.stack((gap / 2 - BRIDGE['clearance'], torch.round((1 + torch.rand(n)) * 10) / 20), -1)
    boxes = torch.rand(n, 3, 3) * torch.tensor([.55, .55, .5]) + torch.tensor([.25, .25, .2])
    layout = sample_bridge_layout(half_a, half_b, gap, bridge_half, boxes, BRIDGE, .8, .5)
    eps = 1e-5
    far, landing, start = layout['far_stage'], layout['landing'], layout['start']
    torch.testing.assert_close(far[:, 0], half_a[:, 0] + gap + half_b[:, 0])
    overlap_low = torch.maximum(-half_a[:, 1], far[:, 1] - half_b[:, 1])
    overlap_high = torch.minimum(half_a[:, 1], far[:, 1] + half_b[:, 1])
    torch.testing.assert_close(landing[:, 0], half_a[:, 0] + gap / 2)
    assert (landing[:, 1] - bridge_half[:, 1] >= overlap_low + BRIDGE['landing_margin'] - eps).all()
    assert (landing[:, 1] + bridge_half[:, 1] <= overlap_high - BRIDGE['landing_margin'] + eps).all()
    end = torch.maximum(half_a[:, 1], start[:, 1].sign() * far[:, 1] + half_b[:, 1])
    assert (start[:, 1].abs() - bridge_half[:, 1] >= end + BRIDGE['approach_distance'][0] - eps).all()
    objects, radius = layout['objects'], boxes[..., :2].norm(dim=-1) / 2
    near_edge = (half_a[:, 0] + gap)[:, None]
    assert (objects[..., 0] - radius >= near_edge + BRIDGE['object_band'][0] - eps).all()
    assert ((objects[..., 1] - far[:, 1:]).abs() + radius <= half_b[:, 1:] + eps).all()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert ((objects[:, i] - objects[:, j]).norm(dim=-1) >= radius[:, i] + radius[:, j]).all()
    humans = layout['humans']
    assert (humans.abs() <= half_a[:, None] - .8 + eps).all()
    assert ((humans[:, 0] - humans[:, 1]).norm(dim=-1) >= 1 - eps).all()


def test_trajectory_schedule_and_force():
    start, goal = torch.tensor([[0., -5., .745]]), torch.tensor([[0., 1., .745]])
    for elapsed, expected in ((-1., start), (4., goal)):
        pos, vel = bridge_trajectory(start, goal, torch.tensor([elapsed]), torch.tensor([4.]))
        torch.testing.assert_close(pos, expected)
        torch.testing.assert_close(vel, torch.zeros(1, 3))
    distance, arrival = torch.tensor([6., 6., 6.]), torch.tensor([10., 2., 30.])
    duration, wait = bridge_schedule(distance, arrival, BRIDGE['speed_range'], 1 / 30)
    torch.testing.assert_close(wait + duration, arrival)
    assert wait[1] < 0 < wait[2]
    state = torch.zeros(1, 13)
    state[0, :3], state[0, 6] = goal, 1
    force = bridge_control(goal + torch.tensor([[0., 10., 0.]]), torch.zeros(1, 3), state, torch.tensor([1000.]), 2., 1., 6.)
    torch.testing.assert_close(force, torch.tensor([[0., 6000., 0.]]))


def test_four_family_amp_labels():
    obs = torch.randn(6, 10, 129)
    family = torch.tensor([0, 1, 2, 3, 3, 0])[:, None].expand(-1, 10)
    labelled = append_family(obs, family, 4)
    assert labelled.shape == (6, 10, 133)
    flat = labelled.flatten(1)
    kept = preserve_amp_labels(flat, flat * 0, 10, 4).reshape(6, 10, 133)
    torch.testing.assert_close(kept[..., -4:], labelled[..., -4:])
    matched = sample_family_matched(flat, flat[[3, 0]], 10, num_families=4)
    assert matched.reshape(2, 10, 133)[:, 0, -4:].argmax(-1).tolist() == [3, 0]


@pytest.mark.skipif(not (ROOT/EXPERIMENT['transfer']['checkpoint']).is_file(), reason='PUSH raw checkpoint missing')
def test_push_checkpoint_transfer_and_predecessor_relations():
    torch.set_num_threads(1)
    builder = BridgeBuilder()
    builder.load(TRAIN['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(636,), amp_input_shape=(1330,),
        value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=ENV['relationReward']['mode'], relation_graph_spec=ENV['relationGraph'], device='cpu')
    model = nn.Module(); model.a2c_network = net
    saved = torch.load(ROOT/EXPERIMENT['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    report = transfer_stage1(model, saved)
    assert len(report['copied']) == len(saved['model'])
    torch.testing.assert_close(net.actor_encoder.edge_encoder.category_embed.weight,
                               saved['model']['a2c_network.actor_encoder.edge_encoder.category_embed.weight'])
    graph = compose_canonical_graph(torch.tensor([[1, 2], [2, 2]]), num_objects=4)
    poses = torch.zeros(2, 8, 7); poses[..., 6] = 1.
    scene = torch.cat((torch.randn(2, 568), poses.flatten(1), task_graph_packet(graph, 2)), -1)
    obs = torch.cat((scene, torch.tensor([[-1., -1.], [1., -1.]])), -1)
    mu, _ = net.eval_actor(obs)
    torch.testing.assert_close(mu, net.action_head(net.actor_encoder(scene)).reshape(4, 32), rtol=1e-5, atol=1e-5)
    (mu.square().mean() + net.eval_critic(obs).square().mean()).backward()
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert net.coordination.blocks[0].attention.out_proj.weight.grad.abs().sum() > 0
    coordination = net.coordination
    coordination.shuffle_human_order = False
    with torch.no_grad():
        for block in coordination.blocks:
            nn.init.normal_(block.attention.out_proj.weight)
    humans, packet = torch.randn(2, 2, 64), task_graph_packet(graph, 2)
    relations = predecessor_relations(obs[:, -2:], 2)
    base = coordination(humans, None, packet, None, relations=relations).detach()
    with torch.no_grad():
        for block in coordination.blocks:
            block.relation_bias[:, 2] += 5.
    moved = coordination(humans, None, packet, None, relations=relations).detach()
    torch.testing.assert_close(moved[0], base[0])
    assert not torch.allclose(moved[1], base[1])


def test_push_config_contract_and_bank():
    validate_env(PUSH_ENV)
    for key in ('relationReward', 'box', 'templateRsi', 'skillDiscProb', 'eval', 'before', 'numAgents', 'numObjects'):
        assert PUSH_ENV[key] == ENV[key]
    assert PUSH_ENV['episodeLength'] == 1200
    load_bank(PUSH_ENV['bridge']['push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    contract = checkpoint_contract(PUSH_ENV)
    assert len(contract['rsi_bank_sha256']) == 64
    check_checkpoint({'bridge_contract': contract}, PUSH_ENV)
    with pytest.raises(ValueError, match='Bridge checkpoint'):
        check_checkpoint({'bridge_contract': checkpoint_contract(ENV)}, PUSH_ENV)
    bad = deepcopy(PUSH_ENV)
    bad['bridge']['scene_probabilities']['present'] = .3
    with pytest.raises(ValueError):
        validate_env(bad)


def test_push_scenes_graph_and_dependency():
    torch.manual_seed(2)
    n = 4096
    scene = sample_scenes(n, 'random_scenario', PUSH_ENV['bridge']['scene_probabilities'], 'cpu')
    fractions = torch.bincount(scene, minlength=3).float() / n
    assert torch.allclose(fractions, torch.tensor([.6, .2, .2]), atol=.02)
    assert (sample_scenes(4, 'decoy_push', PUSH_ENV['bridge']['scene_probabilities'], 'cpu') == SCENES.index('independent')).all()
    crosser = torch.randint(2, (n,))
    pusher = 1 - crosser
    templates = sample_templates(n, 'random_scenario', BRIDGE['task_probabilities'], 'cpu')
    rows = torch.arange(n)
    pushing = scene != SCENES.index('present')
    templates[rows, pusher] = torch.where(pushing, 3, templates[rows, pusher])
    graph, predecessor = push_episode_graph(templates, crosser, scene, 4)
    packet = task_graph_packet(graph, n).reshape(n, 2, 5).long()
    assert torch.equal(packet[rows, pusher, 1] == 4, pushing)
    assert torch.equal(packet[rows, crosser, 1], templates[rows, crosser] - 1)
    assert torch.equal(packet[rows[pushing], pusher[pushing], 3], 2 + pusher[pushing])
    dependent = scene == 0
    assert torch.equal(graph.prereq_mask.flatten(1).any(-1), dependent)
    relations = predecessor_relations(predecessor.float(), 2)
    assert torch.equal(relations[rows, crosser, pusher] == 2, dependent)
    assert (relations[rows, pusher, crosser] == 0).all()
    edge = graph.prereq_mask.any(-1).long().argmax(-1)
    assert (graph.edge_owner[rows[dependent], edge[dependent]] == crosser[dependent]).all()


def test_token_mapping_with_bridge_and_decoy_cubes():
    torch.manual_seed(3)
    primary = torch.tensor([[2, 8], [9, 5], [1, 2]]).repeat(512, 1)
    other = torch.tensor([9, 8, 8]).repeat(512)
    mapping = token_mapping(primary, other, 8)
    assert torch.equal(mapping[:, :2], primary)
    tail = mapping[:, 2:]
    assert ((tail == other[:, None]).sum(-1) == 1).all()
    distractor = torch.where(tail[:, 0] == other, tail[:, 1], tail[:, 0])
    assert ((distractor < 8) & (distractor != primary[:, 0]) & (distractor != primary[:, 1])).all()


def test_push_contact_requires_both_hands_on_the_back_face():
    box = torch.tensor([[0., 0., .75, 0., 0., 0., 1.] + [0.] * 6])
    size = torch.tensor([[1.5, 1.5, 1.5]])
    hands = torch.tensor([[[.2, -.8, 1.], [-.2, -.8, 1.]]])
    push = torch.tensor([[[0., -5., 0.], [0., -5., 0.]]])
    phi, hit, point, normal = contact_geometry(hands, box, size, push, torch.tensor([[0., 8., 0.]]))
    assert hit.item() and phi.item() > .9
    torch.testing.assert_close(point, torch.tensor([[0., -.75, .75]]))
    _, hit, _, _ = contact_geometry(hands + torch.tensor([0., -.5, 0.]), box, size, push, torch.tensor([[0., 8., 0.]]))
    assert not hit.item()
    _, hit, _, _ = contact_geometry(hands, box, size, push, torch.tensor([[0., -8., 0.]]))
    assert not hit.item()
