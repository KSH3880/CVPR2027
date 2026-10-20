from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from utils.before_spec import sample_graph, validate_env, eligible_families, checkpoint_contract
from utils.task_role_spec import task_graph_packet
from utils.edge_ontop_spec import permute_graph
from env.tasks.multi_agent.before_reward import BeforeRuntime
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from learning.multi_agent.human_coordination import human_relations, HumanCoordination
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from utils.stage2_spec import check_checkpoint

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_before.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage2_before.yaml').read_text())
RANGES = ENV['box']['build']['taskSizeRanges']


def graph(preset, n=64):
    return sample_graph(torch.full((n, 4, 3), .4), RANGES, preset)


def test_base_settings_and_checkpoint_isolation():
    validate_env(ENV)
    base = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_shared_full_success.yaml').read_text())['env']
    for key in ('relationReward', 'box', 'templateRsi', 'skillDiscProb', 'sizeAwareRsi', 'plane'):
        assert ENV[key] == base[key]
    assert ENV['numEnvs'] == 1024 and ENV['numObjects'] == 8
    assert 'sharedSupport' not in ENV and 'stage2' not in ENV
    check_checkpoint({'before_contract': checkpoint_contract(ENV)}, ENV)
    with pytest.raises(ValueError, match='BEFORE checkpoint'):
        check_checkpoint({}, ENV)
    bad = deepcopy(ENV)
    bad['before']['early_penalty'] = .1
    with pytest.raises(ValueError):
        validate_env(bad)


def test_ground_config_isolation_and_checkpoint_contract():
    from utils.stage1_v2_fixed import uses_fixed_platforms
    ground = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_before_ground.yaml').read_text())['env']
    validate_env(ground)
    assert uses_fixed_platforms(ground)
    assert ground['box']['reset']['randomHeight'] is False
    assert ground['box']['reset']['randomHeightProb'] == 0.
    assert ground['box']['reset']['platformMode'] == 'ground_only'
    comparable = deepcopy(ground)
    for key in ('randomHeight', 'randomHeightProb', 'platformMode'):
        comparable['box']['reset'][key] = ENV['box']['reset'][key]
    assert comparable == ENV
    for source, target in ((ground, ENV), (ENV, ground)):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    bad = deepcopy(ground)
    bad['box']['reset']['randomHeightProb'] = .5
    with pytest.raises(ValueError, match='Ground BEFORE'):
        validate_env(bad)


def test_ground_reset_preserves_floor_boxes_and_goals():
    from types import SimpleNamespace
    from utils.stage1_platforms import reset_stage1_platforms
    boxes = torch.randn(3, 8, 13)
    boxes[..., 2] = .2
    goals = torch.randn(3, 4, 3)
    goals[..., 2] = .2
    task = SimpleNamespace(_ground_only=True, _box_states=boxes.clone(), _tar_pos=goals.clone(),
        _platform_pos=torch.ones(3, 4, 3), _tar_platform_pos=torch.ones(3, 4, 3),
        _platform_default_pos=torch.full((3, 4, 3), -20.),
        _tar_platform_default_pos=torch.full((3, 4, 3), -20.))
    reset_stage1_platforms(task, torch.tensor([0, 2]))
    torch.testing.assert_close(task._box_states, boxes)
    torch.testing.assert_close(task._tar_pos, goals)
    for value in (task._platform_pos, task._tar_platform_pos):
        assert (value[[0, 2]] == -20).all()
        assert (value[1] == 1).all()


@pytest.mark.parametrize('preset', ['random_scenario', 'before_sit', 'before_climb', 'before_stack',
                                  'one_pair_stack', 'all_solo', 'sit', 'holding_ontop'])
def test_pair_roles_dependency_and_permutations(preset):
    g, pre, templates = graph(preset)
    packet = task_graph_packet(g, 64)
    relation = human_relations(packet, 4, before=True)
    row, follower = (pre >= 0).nonzero(as_tuple=True)
    first = pre[row, follower]
    assert (pre[row, first] == -1).all()
    assert (templates[row, first] == 3).all()
    assert (relation[row, follower, first] == 2).all()
    assert (relation[row, first, follower] == 0).all()
    assert int((relation == 2).sum()) == len(row)
    assert torch.equal(g.prereq_mask.sum((1, 2)), (pre >= 0).sum(-1))
    shuffled = permute_graph(g, torch.rand(64, 8).argsort(-1))
    assert torch.equal(packet, task_graph_packet(shuffled, 64))
    shuffled_packet = packet.reshape(64, 4, 5)[:, [2, 0, 3, 1]].flatten(1)
    assert torch.equal(relation, human_relations(shuffled_packet, 4, before=True))


def test_sampling_frequency_and_size_conditioning():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    n = 30000
    sizes = torch.randint(4, 13, (n, 4, 3))/20.
    _, pre, templates = sample_graph(sizes, RANGES)
    frequency = torch.bincount((pre >= 0).sum(-1), minlength=3)/n
    torch.testing.assert_close(frequency, torch.tensor([.008, .352, .64]), rtol=0, atol=.012)
    row, human = (pre >= 0).nonzero(as_tuple=True)
    family = torch.where(templates[row, human] == 1, 0, torch.where(templates[row, human] == 2, 1, 2))
    assert eligible_families(sizes, RANGES)[row, pre[row, human], family].all()


def test_compact_tokens_include_real_distractors():
    from utils.before_spec import pack_objects
    parents = torch.tensor([[0, 5, 2, 3, 12, 13, 14, 15], [4, 5, 6, 7, 12, 13, 14, 15]])
    valid = torch.tensor([[True, False, True, True, False, False, True, False],
                          [True, True, True, True, False, False, False, False]])
    mapping, inverse, count = pack_objects(parents, valid)
    for row in range(2):
        assert len(mapping[row].unique()) == 8
        assert mapping[row].min() >= 0 and mapping[row].max() < 16
        torch.testing.assert_close(mapping[row, inverse[row,valid[row]]], parents[row,valid[row]])
        assert not torch.isin(mapping[row,count[row]:], parents[row,valid[row]]).any()


@pytest.mark.parametrize('humans,objects', [(1, 1), (3, 3), (5, 5), (6, 8), (6, 12), (2, 11)])
def test_variable_counts_and_object_budget(humans, objects):
    from utils.before_spec import pack_objects, task_objects
    env = deepcopy(ENV)
    env['numAgents'], env['numObjects'] = humans, objects
    validate_env(env)
    torch.manual_seed(47)
    sizes = torch.full((256, humans, 3), .4)
    g, pre, templates = sample_graph(sizes, RANGES, num_objects=objects)
    valid = task_objects(pre, templates)
    assert (valid.sum(-1) <= objects).all()
    people = torch.arange(humans)[None].expand(256, -1)
    source = people+torch.where(templates == 1, humans, torch.where(templates == 2, 2*humans, 0))
    parents = torch.cat((source, 3*humans+people), -1)
    mapping, inverse, count = pack_objects(parents, valid, objects)
    for row in range(256):
        assert len(mapping[row].unique()) == objects
        torch.testing.assert_close(mapping[row,inverse[row,valid[row]]], parents[row,valid[row]])
    row, follower = (pre >= 0).nonzero(as_tuple=True)
    assert (pre[row,pre[row,follower]] == -1).all()
    assert (g.prereq_mask.sum((1, 2)) == (pre >= 0).sum(-1)).all()
    for r in range(256):
        leaders = pre[r,pre[r] >= 0]
        assert len(leaders.unique()) == len(leaders)
    if objects == humans:
        _, pre, templates = sample_graph(sizes, RANGES, 'all_solo', objects)
        assert (pre == -1).all() and not (templates == 4).any()


def test_budget_rejection_redraws_only_overfull_environments(monkeypatch):
    from utils import before_spec
    calls = []
    def roles(sizes, ranges, preset):
        calls.append(len(sizes))
        pre = torch.full(sizes.shape[:2], -1, dtype=torch.long)
        templates = torch.full_like(pre, 3)
        if len(calls) == 1:
            templates[0] = 4
        else:
            templates[:] = 1
        return pre, templates
    monkeypatch.setattr(before_spec, 'sample_roles', roles)
    _, _, templates = sample_graph(torch.full((3, 3, 3), .4), RANGES, num_objects=3)
    assert calls == [3, 1]
    assert (templates[0] == 1).all() and (templates[1:] == 3).all()


def test_impossible_counts_and_fixed_presets():
    with pytest.raises(ValueError, match='numObjects'):
        sample_graph(torch.full((1, 3, 3), .4), RANGES, num_objects=2)
    with pytest.raises(ValueError, match='two objects'):
        sample_graph(torch.full((1, 3, 3), .4), RANGES, 'holding_ontop', 3)
    with pytest.raises(ValueError, match='two humans'):
        sample_graph(torch.full((1, 1, 3), .4), RANGES, 'before_stack', 2)


def test_odd_before_presets_leave_one_solo():
    _, pre, _ = sample_graph(torch.full((32, 5, 3), .4), RANGES, 'before_stack', 5)
    assert ((pre >= 0).sum(-1) == 2).all()
    _, pre, _ = sample_graph(torch.full((32, 5, 3), .4), RANGES, 'one_pair_stack', 5)
    assert ((pre >= 0).sum(-1) == 1).all()


def test_before_training_diagnostics_are_scalars():
    from types import SimpleNamespace
    from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
    task = SimpleNamespace(_edge_metric_sums={'before/gate': torch.tensor(3.)},
        _edge_metric_denominators={'before/gate': torch.tensor(4.)},
        _edge_metric_steps=1, _edge_episode_sums=torch.zeros(6),
        _consume_sampling_diagnostics=lambda: {})
    result = EdgeContextTaskMixin.consume_relation_diagnostics(task)
    assert result['before/gate'] == .75
    assert task._edge_metric_sums == {}


@pytest.mark.parametrize('preset', ['before_sit', 'before_climb', 'before_stack', 'all_solo'])
def test_current_gate_progress_and_solo_equivalence(preset):
    g, _, _ = graph(preset, 8)
    runtime = BeforeRuntime(8, g, ENV['relationReward'], 'cpu')
    baseline = Stage1ContextRuntime(8, g, ENV['relationReward'], 'cpu')
    phi, progress = torch.full((8, 8), .95), torch.full((8, 8), .4)
    zero = torch.zeros_like(phi)
    dependent = g.prereq_mask.any(-1)
    for distance, opened in ((.201, False), (.199, True), (.201, False)):
        phi[g.edge_relation == 7] = torch.exp(torch.tensor(-10*distance**2))
        result = runtime.step(phi, progress, zero, zero, zero)
        assert (result['prerequisite_gate'][dependent] == opened).all()
        if not opened:
            assert not result['state_component'][dependent].any()
            assert not result['success_component'][dependent].any()
            torch.testing.assert_close(result['progress_component'][dependent], progress[dependent]*.2)
        if preset == 'all_solo':
            expected = baseline.step(phi, progress, zero, zero, zero)
            for key in ('own_success', 'total', 'reward_saturated', 'paired_placement_success'):
                torch.testing.assert_close(result[key], expected[key])


def test_frozen_stage1_transfer_and_before_gradients():
    torch.set_num_threads(1)
    builder = BeforeBuilder()
    builder.load(TRAIN['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(1268,), amp_input_shape=(1320,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=ENV['relationReward']['mode'], relation_graph_spec=ENV['relationGraph'], device='cpu')
    model = nn.Module(); model.a2c_network = net
    saved = torch.load(ROOT/'stage1/Stage1V2Fixed_00030500.pth', map_location='cpu', weights_only=False)
    report = transfer_stage1(model, saved)
    assert len(report['copied']) == len(saved['model'])
    g, _, _ = graph('before_stack', 2)
    poses = torch.zeros(2, 16, 7); poses[..., 6] = 1.
    obs = torch.cat((torch.randn(2, 1136), poses.flatten(1), task_graph_packet(g, 2)), -1)
    mu, _ = net.eval_actor(obs)
    torch.testing.assert_close(mu, net.action_head(net.actor_encoder(obs)).reshape(8, 32), rtol=1e-5, atol=1e-5)
    (mu.square().mean()+net.eval_critic(obs).square().mean()).backward()
    assert net.coordination.before and net.critic_coordination.before
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert net.coordination.blocks[0].attention.out_proj.weight.grad.abs().sum() > 0
    coordination = HumanCoordination(before=True).eval()
    humans = torch.randn(2, 4, 64)
    packet = task_graph_packet(g, 2)
    torch.manual_seed(5); coordination.shuffle_human_order = False
    expected = coordination(humans, None, packet, None)
    coordination.shuffle_human_order = True
    torch.testing.assert_close(coordination(humans, None, packet, None), expected, atol=2e-6, rtol=1e-5)
