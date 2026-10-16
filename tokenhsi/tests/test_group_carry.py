from copy import deepcopy

import pytest
import torch

from utils.group_carry_spec import (GROUP_SPEC, sample_graph, sample_groups,
    physical_mapping, owner_tasks, entity_validity)
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.human_coordination import human_relations
from learning.multi_agent.grasp_view import make_grasp_views, initial_grasp_sides
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from sa_helpers import configs


def scene(preset='mixed_groups', n=8):
    graph = sample_graph(n, GROUP_SPEC, preset=preset)
    obs = torch.zeros(n, 1268)
    obs[:, :892] = torch.randn(n, 892)*.1
    objects = obs[:, 892:1132].reshape(n, 8, 30)
    signs = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    objects[..., 6:] = (signs*torch.tensor([.26, .4, .2])).flatten()
    poses = obs[:, 1136:1248].reshape(n, 16, 7)
    poses[..., 6] = 1.
    poses[..., :2] = torch.randn(n, 16, 2)
    obs[:, -20:] = task_graph_packet(graph, n)
    return obs, graph


def network(grasp=False):
    name = 'stage2_grasp_at20_4agent' if grasp else 'stage2_at20_4agent'
    env, train = configs(name)
    builder = AMPMultiAgentBuilder()
    builder.load(train['params']['network'])
    return builder.build('amp', actions_num=32, input_shape=(5072 if grasp else 1268,),
        amp_input_shape=(1320,), value_size=1, num_agents=4, num_objects=8,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=10.,
        relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=GROUP_SPEC, device='cpu')


def test_group_distribution_and_exchangeability():
    torch.manual_seed(421)
    groups = sample_groups(60000)
    joint = (groups[:, :, None] == groups[:, None]).sum(-1) == 2
    freq = torch.bincount(joint.sum(-1)//2, minlength=3)/len(groups)
    torch.testing.assert_close(freq, torch.tensor([.008, .352, .64]), atol=.008, rtol=0)
    assert (joint.float().mean(0)-.816).abs().max() < .008
    pair = (groups[:, :, None] == groups[:, None]).float().mean(0)
    off_diagonal = pair[~torch.eye(4, dtype=torch.bool)]
    assert off_diagonal.max()-off_diagonal.min() < .012


@pytest.mark.parametrize('preset, pairs, objects', [('all_solo_ontop', 0, 8),
    ('one_pair_ontop', 1, 6), ('two_pairs_ontop', 2, 4)])
def test_object_and_relation_ownership(preset, pairs, objects):
    graph = sample_graph(100, GROUP_SPEC, preset=preset)
    packet = task_graph_packet(graph, 100)
    mapping, groups, joint = physical_mapping(packet)
    valid = entity_validity(packet, 4, 8)[:, 4:12]
    assert (valid.sum(-1) == objects).all()
    assert (joint.sum(-1) == pairs*2).all()
    for row in range(100):
        used = mapping[row, valid[row]]
        assert len(used.unique()) == objects
    relation = human_relations(packet, 4)
    assert ((relation == 1).sum(-1) == 1).all()
    assert torch.equal((relation == 2).any(-1), joint)
    assert torch.equal(owner_tasks(packet.reshape(100, 4, 5)[:, [2, 0, 3, 1]].flatten(1), 4), packet.reshape(100, 4, 5))


@pytest.mark.parametrize('grasp', [False, True])
def test_forward_permutations_and_padding(grasp):
    torch.manual_seed(3)
    obs, graph = scene()
    net = network(grasp).eval()
    sides = initial_grasp_sides(obs, 4, 8)
    views = make_grasp_views(obs, sides, .05) if grasp else obs
    mu, _ = net.eval_actor(views)
    value = net.eval_critic(views)
    assert mu.shape == (32, 32) and torch.isfinite(mu).all() and torch.isfinite(value).all()
    changed = obs.clone()
    changed[:, -20:] = changed[:, -20:].reshape(-1, 4, 5)[:, [2, 0, 3, 1]].flatten(1)
    if grasp:
        torch.testing.assert_close(make_grasp_views(changed, sides, .05)[:, :1248], views[:, :1248])
    valid = entity_validity(obs[:, -20:], 4, 8)
    nodes = changed[:, 892:1132].reshape(-1, 8, 30)
    nodes[~valid[:, 4:12]] = torch.randn_like(nodes[~valid[:, 4:12]])*100
    changed = make_grasp_views(changed, sides, .05) if grasp else changed
    other, _ = net.eval_actor(changed)
    torch.testing.assert_close(mu, other, atol=2e-5, rtol=2e-5)
    raw = views.reshape(-1, 1268)
    encoder = net.actor_encoder
    a = encoder(raw, token_order=torch.arange(16))
    b = encoder(raw, token_order=torch.randperm(16))
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_group_grasp_only_own_payload_and_rigid_velocity():
    obs, _ = scene()
    tasks = owner_tasks(obs[:, -20:], 4).long()
    sides = initial_grasp_sides(obs, 4, 8)
    relation = human_relations(obs[:, -20:], 4)
    joint = (relation == 2).any(-1)
    objects = obs[:, 892:1132].reshape(-1, 8, 30)
    objects[..., 5] = 2.
    views = make_grasp_views(obs, sides, .05).reshape(-1, 4, 1268)
    for row in range(len(obs)):
        for human in range(4):
            if not joint[row, human]:
                torch.testing.assert_close(views[row, human], obs[row])
                continue
            partner = (relation[row, human] == 2).nonzero().item()
            assert sides[row, human] == -sides[row, partner]
            payload = tasks[row, human, 3].item()-4
            delta = views[row, human]-obs[row]
            assert delta.count_nonzero() == 2
            assert abs(delta[892+payload*30].item()+.7*sides[row, human].item()) < 1e-6


def test_gate_and_cpa_isolation():
    from env.tasks.multi_agent.group_carry_reward import apply_group_geometry, group_cpa_release
    from env.tasks.multi_agent.joint_carry_reward import JointCarryRuntime, cpa_collision_penalty
    graph = sample_graph(1, GROUP_SPEC, preset='two_pairs_at')
    packet = task_graph_packet(graph, 1)
    _, groups, joint = physical_mapping(packet)
    pair_leaders = groups.unique().tolist()
    boxes = torch.zeros(1, 8, 13); boxes[..., 6] = 1
    sizes = torch.tensor([.52, .8, .4]).expand(1, 8, 3).clone()
    hands = torch.full((1, 4, 2, 3), 10.)
    first = (groups[0] == pair_leaders[0]).nonzero().flatten()
    hands[0, first[0]] = torch.tensor([0., -.35, 0.])
    hands[0, first[1]] = torch.tensor([0., .35, 0.])
    phi = torch.ones(1, 8)
    diag = {'target': torch.zeros(1, 8, 3), 'distance': torch.zeros(1, 8)}
    config = configs('stage2_at20')[0]['relationReward']
    phi, gate = apply_group_geometry(phi, diag, hands, boxes, sizes, graph, config)
    assert torch.equal(gate[0], groups[0] == pair_leaders[0])
    runtime = JointCarryRuntime(1, graph, config, 'cpu')
    active = runtime.active_edges(phi, graph, gate)
    placements = graph.edge_relation == 7
    assert (active & placements).sum() == 2
    roots = torch.zeros(1, 4, 3)
    release = group_cpa_release(roots, boxes, sizes, graph, 1.)
    assert torch.equal(release, human_relations(packet, 4) == 2)
    other = (groups[0] != pair_leaders[0]).nonzero().flatten()[0]
    roots[0, other, 0] = .5
    vel = torch.zeros_like(roots); vel[0, other, 0] = -1.
    penalty = cpa_collision_penalty(roots, vel, .7, .99, 1/30, release)
    assert penalty[0, first[0]] > 0


def test_group_checkpoints_are_separate_from_two_humans():
    from utils.joint_carry_amp import validate_training, check_training_checkpoint
    from utils.joint_carry_spec import validate_env
    for grasp in (False, True):
        name = 'stage2_grasp_at20' if grasp else 'stage2_at20'
        env, train = configs(name+'_4agent')
        validate_env(env)
        validate_training(env, train)
        weights = dict(joint_carry_grasp_view=env.get('jointCarryGraspView'),
            joint_carry_loco_platform=env['jointCarryRsi']['locoPlatform'],
            joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
            joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
            joint_carry_training=env['jointCarryTraining'])
        check_training_checkpoint(weights, env)
        with pytest.raises(ValueError, match='contract differs'):
            check_training_checkpoint(weights, configs(name)[0])


def test_same_anchor_does_not_open_pair_gate():
    from env.tasks.multi_agent.group_carry_reward import apply_group_geometry
    graph = sample_graph(8, GROUP_SPEC, preset='two_pairs_at')
    boxes = torch.zeros(8, 8, 13); boxes[..., 6] = 1
    sizes = torch.tensor([.52, .8, .4]).expand(8, 8, 3)
    hands = torch.zeros(8, 4, 2, 3); hands[..., 1] = -.35
    diag = {'target': torch.zeros(8, 8, 3), 'distance': torch.zeros(8, 8)}
    phi, gate = apply_group_geometry(torch.zeros(8, 8), diag, hands, boxes, sizes, graph,
        configs('stage2_at20')[0]['relationReward'])
    assert not gate.any()
    assert (phi[graph.edge_relation == 6] >= configs('stage2_at20')[0]['relationReward']['satisfaction_threshold']).all()


@pytest.mark.parametrize('grasp', [False, True])
def test_four_human_action_permutation(grasp):
    obs, _ = scene()
    net = network(grasp).eval()
    order = torch.tensor([2, 0, 3, 1])
    inverse = order.argsort()
    moved = obs.clone()
    moved[:, :892] = obs[:, :892].reshape(-1, 4, 223)[:, order].flatten(1)
    poses = moved[:, 1136:1248].reshape(-1, 16, 7)
    poses[:, :4] = poses[:, :4][:, order].clone()
    packet = moved[:, -20:].reshape(-1, 4, 5)
    packet[..., 2] = inverse[packet[..., 2].long()]
    if grasp:
        obs = make_grasp_views(obs, initial_grasp_sides(obs, 4, 8), .15)
        moved = make_grasp_views(moved, initial_grasp_sides(moved, 4, 8), .15)
    mu = net.eval_actor(obs)[0].reshape(-1, 4, 32)
    shuffled = net.eval_actor(moved)[0].reshape(-1, 4, 32)
    torch.testing.assert_close(shuffled, mu[:, order], atol=2e-5, rtol=2e-5)


def test_goals_clear_other_humans_and_goals():
    from types import SimpleNamespace
    from env.tasks.multi_agent.group_carry_reset import scene_invalid
    graph = sample_graph(1, GROUP_SPEC, preset='all_solo_at')
    roots = torch.tensor([[[0., 0., 1.], [5., 0., 1.], [0., 5., 1.], [5., 5., 1.]]])
    boxes = torch.zeros(1, 8, 13); boxes[..., 6] = 1.
    boxes[:, :4, :2] = roots[..., :2]+torch.tensor([1.5, 0.])
    boxes[..., 2] = .15
    targets = roots.clone(); targets[..., 1] += 2.5; targets[..., 2] = .15
    task = SimpleNamespace(num_envs=1, device='cpu', relation_runtime=SimpleNamespace(graph=graph),
        _group_leader=torch.arange(4)[None], _box_states=boxes, _box_size=torch.full((1, 8, 3), .3),
        _kinematic_humanoid_rigid_body_states=roots[:, :, None], _tar_pos=targets,
        _env_origins=torch.zeros(1, 3), cfg={'env': {'envSpacing':10}},
        _logical_box_values=lambda value, ids: value[ids])
    assert not scene_invalid(task, torch.tensor([0])).any()
    task._tar_pos[0, 0, :2] = roots[0, 1, :2]
    assert scene_invalid(task, torch.tensor([0])).all()
    task._tar_pos[0, 0, :2] = targets[0, 1, :2]
    assert scene_invalid(task, torch.tensor([0])).all()


def test_unused_physical_actors_are_separated_and_masked():
    from types import SimpleNamespace
    from env.tasks.multi_agent.group_carry_reset import park_unused_boxes
    graph = sample_graph(12, GROUP_SPEC)
    packet = task_graph_packet(graph, 12)
    mapping = physical_mapping(packet)[0]
    states = torch.randn(12, 12, 13)
    before = states.clone()
    origins = torch.randn(12, 3); origins[:, 2] = 0.
    task = SimpleNamespace(num_envs=12, device='cpu', relation_runtime=SimpleNamespace(graph=graph),
        _box_states=states, _box_size=torch.full((12, 12, 3), .5),
        _logical_box_order=mapping, _env_origins=origins)
    ids = torch.arange(0, 12, 2)
    park_unused_boxes(task, ids)
    torch.testing.assert_close(states[1::2], before[1::2])
    active = torch.zeros(12, 12, dtype=torch.bool).scatter_(1, mapping, entity_validity(packet, 4, 8)[:, 4:12])
    for row in ids:
        torch.testing.assert_close(states[row, active[row]], before[row, active[row]])
        parked = states[row, ~active[row]]
        assert ((parked[:, :2]-origins[row, :2]).norm(dim=-1) > 20.).all()
        distance = torch.cdist(parked[:, :2], parked[:, :2])
        distance.fill_diagonal_(float('inf'))
        assert (distance >= 2.-1e-5).all()


def test_four_agent_experiments_differ_only_in_grasp_view():
    plain_env, plain_train = configs('stage2_at20_4agent')
    grasp_env, grasp_train = configs('stage2_grasp_at20_4agent')
    grasp_env.pop('jointCarryGraspView')
    assert plain_env == grasp_env
    grasp_train['params']['network']['coordination'].pop('grasp_view')
    grasp_train['params']['config']['name'] = plain_train['params']['config']['name']
    assert plain_train == grasp_train
