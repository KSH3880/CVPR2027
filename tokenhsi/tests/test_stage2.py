from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.unified_training import validate_unified_env
from utils import stage2_spec as spec
from utils import coupled_push_spec as coupled
from utils.before_object_alias import alias_layout, encoder_packet, goal_aligned_scene
from utils.task_role_spec import task_graph_packet
from utils.edge_ontop_spec import select_graph, validate_graph
from learning.multi_agent.human_coordination import stage2_human_relations
from learning.multi_agent.stage2_push_network import transfer_stage1
from learning.multi_agent.stage2_relation_fusion import Stage2RelationFusion, Stage2FusionCoordinationBlock
from learning.multi_agent.coupled_relation_fusion import CoupledRelationFusion, FusionCoordinationBlock
from env.tasks.multi_agent.stage2_reward import Stage2Runtime
from env.tasks.multi_agent.coupled_push_reward import CoupledPushRuntime
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.stage2_wait import Stage2Wait
from env.tasks.multi_agent.stage2_layout import resample_before_references
from utils.before_goal_sampling import sample_layout
from test_stage2_push_common import build, _common_observation

ENV = native_config('stage2')['env']


def sampled(preset, count=128):
    domain, groups, predecessor, templates = spec.sample_episode(torch.full((count, 4, 3), .4),
        ENV['box']['build']['taskSizeRanges'], ENV['relationGraph']['template_probabilities'], preset)
    graph = spec.compose_graph(templates, groups, predecessor, 8)
    return domain, groups, predecessor, templates, graph


def test_config_contract_and_entity_limits():
    validate_unified_env(ENV)
    validate_experiment('stage2', ENV, native_config('stage2', True))
    assert experiment_horizon('stage2') == 1200
    assert ENV['relationReward']['success']['at_z_tolerance'] == .001
    assert ENV['stage2']['before_at_z_tolerance'] == .2
    for key, value in [('numAgents', 5), ('numObjects', 3)]:
        env = deepcopy(ENV)
        env[key] = value
        with pytest.raises(ValueError):
            validate_unified_env(env)
    bad = deepcopy(ENV)
    bad['before']['waiting_penalty']['coefficient'] = 2.
    with pytest.raises(ValueError):
        validate_unified_env(bad)
    saved = {'coupled_push_contract': coupled.checkpoint_contract(ENV)}
    coupled.check_checkpoint(saved, ENV)
    with pytest.raises(ValueError):
        coupled.check_checkpoint(saved, native_config('stage2_coupled_push')['env'])


@pytest.mark.parametrize('preset', spec.PRESETS)
def test_graph_relations_physics_and_alias_budget(preset):
    _, groups, predecessor, templates, graph = sampled(preset)
    mapping, primary, task_slots = spec.physical_mapping(templates, groups, predecessor, graph)
    packet = task_graph_packet(graph, len(templates))
    relations = stage2_human_relations(packet, groups)
    dependent, _, alias, _ = alias_layout(packet, 4, 8)
    valid, task, actor, payload, target = packet.reshape(-1, 4, 5).long().unbind(-1)
    used = ((payload[..., None] == torch.arange(4, 12)) & (task[..., None] >= 2))
    used |= target[..., None] == torch.arange(4, 12)
    rows, record = dependent.nonzero(as_tuple=True)
    assert not used.any(1)[rows, alias[rows, record]-4].any()
    assert mapping.shape == (len(templates), 8) and mapping.max() < 24
    assert task_slots.sum(-1).max() <= 8
    assert torch.equal(mapping[rows, alias[rows, record]-4], mapping[rows, target[rows, record]-4])
    assert torch.equal((relations == 2).sum((1, 2)), (predecessor >= 0).sum(-1))
    assert not ((relations == 3).any(-1) & spec.before_members(predecessor)).any()
    for row in range(4):
        validate_graph(select_graph(graph, row))
    if preset == 'mixed':
        assert spec.before_members(predecessor).sum(-1).eq(2).all()
        assert (templates == 5).sum(-1).eq(2).all()
        assert (relations == 3).sum((1, 2)).eq(2).all()
        physical_groups = primary[:, :, None] == primary[:, None]
        assert physical_groups[relations == 3].all()
    eval_groups = spec.evaluation_groups(packet, groups)
    assert ((coupled.group_sizes(eval_groups) > 1) == (relations == 3).any(-1)).all()


def test_scene_types_follow_equal_weights():
    torch.manual_seed(37)
    domain, _, predecessor, _, _ = sampled('random_scenario', 12000)
    proportions = torch.bincount(domain, minlength=3).float()/len(domain)
    torch.testing.assert_close(proportions, torch.full((3,), 1/3), atol=.015, rtol=0)
    assert (predecessor >= 0).any()


@pytest.mark.parametrize('template', [1, 2, 4])
@pytest.mark.parametrize('near', [False, True])
def test_before_goalroute_preserves_distance_and_near_constraints(template, near):
    torch.manual_seed(19)
    count = 2048
    goal = torch.zeros(count, 2)
    leader_route = torch.linspace(2.1, 10., count)
    templates = torch.full((count,), template)
    roots, payload = sample_layout(goal, leader_route, templates, torch.full((count,), near))
    route = roots.norm(dim=-1) if template != 4 else (
        (roots-payload).norm(dim=-1)+payload.norm(dim=-1))
    assert (roots.norm(dim=-1) >= 1.).all()
    assert (route <= 10.00001).all()
    if near:
        assert (route < leader_route).all()
    if template == 4:
        for values in ((roots-payload).norm(dim=-1), payload.norm(dim=-1)):
            assert ((values >= 1.-1e-5) & (values <= 5.+1e-5)).all()


def test_mixed_virtual_coordinates_only_change_free_alias_xyz():
    _, groups, predecessor, _, graph = sampled('mixed', 8)
    packet = task_graph_packet(graph, 8)
    scene = _common_observation('solo')[:, :-4].expand(8, -1).clone()
    scene[:, -20:] = packet
    dependent, target, alias, _ = alias_layout(packet, 4, 8)
    changed = goal_aligned_scene(scene, 4, 8)
    poses = scene[:, 1136:1248].reshape(8, 16, 7)
    result = changed[:, 1136:1248].reshape_as(poses)
    records = packet.reshape(8, 4, 5).long()
    rows, followers = dependent.nonzero(as_tuple=True)
    actors = records[rows, followers, 2]
    leaders = predecessor[rows, actors]
    leader_record = (records[rows, :, 2] == leaders[:, None]).long().argmax(-1)
    goal = records[rows, leader_record, 4]
    assert torch.equal(result[rows, alias[rows, followers], :3], poses[rows, goal, :3])
    mask = torch.zeros_like(scene, dtype=torch.bool)
    mask[:, 1136:1248].reshape_as(poses)[rows, alias[rows, followers], :3] = True
    assert torch.equal(changed[~mask], scene[~mask])


@pytest.mark.parametrize('kind', [2, 3])
def test_shared_fusion_recovers_single_relation_computation(kind):
    torch.manual_seed(11)
    old = CoupledRelationFusion(64).double()
    new = Stage2RelationFusion(64).double()
    old_block = FusionCoordinationBlock(64, 2, 128, 4, 'zerobg').double()
    new_block = Stage2FusionCoordinationBlock(64, 2, 128).double()
    with torch.no_grad():
        old.mlp[-1].weight.normal_()
        old_block.bias_projection.normal_()
    state = old.state_dict()
    state['embedding'] = state['embedding'].repeat(2, 1)
    new.load_state_dict(state, strict=True)
    new_block.load_state_dict(old_block.state_dict(), strict=True)
    relations = torch.tensor([[[1, 3, 0], [3, 1, 0], [0, 0, 1]]])
    x = torch.randn(1, 3, 64, dtype=torch.double)
    expected = old_block(x, None, relations, torch.tensor(False), old)
    actual = new_block(x, None, torch.where(relations == 3, kind, relations), torch.tensor(False), new)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert new_block.effective_relation_bias(new.embedding)[:, :2].eq(0).all()


def runtime_inputs(graph):
    phi = torch.full_like(graph.edge_relation, .95, dtype=torch.float)
    progress = torch.full_like(phi, .7)
    zero = torch.zeros_like(phi)
    return phi, progress, zero, zero, zero


def test_before_and_solo_coupled_at_height_thresholds_in_one_batch():
    templates = torch.tensor([[3, 1, 3, 3], [3, 4, 5, 5]])
    groups = torch.tensor([[0, 0, 2, 3], [0, 0, 2, 2]])
    predecessor = torch.tensor([[-1, 0, -1, -1], [-1, 0, -1, -1]])
    graph = spec.compose_graph(templates, groups, predecessor, 8, False)
    runtime = Stage2Runtime(2, graph, ENV['relationReward'], 'cpu',
                            contact_weights=ENV['jointPush']['contact_weights'])
    runtime.before[:] = spec.before_members(predecessor)
    phi, progress, z, feet, region = runtime_inputs(graph)
    at = graph.edge_relation == 7
    z = z.clone(); z[at] = .05
    runtime.contact.fill_(True)
    result = runtime.step(phi, progress, z, feet, region)
    before_at = at & runtime.before.gather(1, graph.edge_owner)
    assert result['own_success'][before_at].all()
    assert not result['own_success'][at & ~before_at].any()
    ids = torch.tensor([1])
    runtime.reset(ids, phi[ids], z[ids], feet[ids], region[ids])
    assert runtime.own_success[1, before_at[1]].all()


@pytest.mark.parametrize('preset', ['all_coupled', 'before_stack'])
def test_rewards_recover_original_relation_runtime(preset):
    _, _, predecessor, _, graph = sampled(preset, 16)
    cfg = deepcopy(ENV['relationReward'])
    runtime = Stage2Runtime(16, graph, cfg, 'cpu', contact_weights=[.2]*3)
    runtime.before[:] = spec.before_members(predecessor)
    if preset == 'all_coupled':
        reference = CoupledPushRuntime(16, graph, cfg, 'cpu', contact_weights=[.2]*3)
        runtime.gate[:] = reference.gate[:] = torch.rand_like(runtime.gate.float()) > .3
    else:
        cfg['success']['at_z_tolerance'] = .2
        reference = BeforePushRuntime(16, graph, cfg, 'cpu', contact_weights=[.2]*3)
    values = runtime_inputs(graph)
    actual, expected = runtime.step(*values), reference.step(*values)
    for key in ('own_success', 'state_component', 'progress_component', 'success_component', 'total'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_waiting_penalty_applies_only_to_before_followers_and_latches_release():
    _, _, predecessor, _, graph = sampled('mixed', 3)
    wait = Stage2Wait(graph, 3, ENV['before']['waiting_penalty'], 'cpu')
    raw = torch.zeros_like(graph.edge_valid)
    wait.reset(torch.arange(3), graph, raw)
    roots, velocity = torch.zeros(3, 4, 3), torch.ones(3, 4, 3)
    sizes, objects, goals = torch.ones(3, 8, 3), torch.zeros(3, 8, 13), torch.zeros(3, 4, 3)
    penalty = wait.penalty(roots, velocity, objects, sizes, goals, raw)
    assert torch.equal(penalty > 0, predecessor >= 0)
    raw[wait.rows, wait.predecessor] = True
    assert not wait.penalty(roots, velocity, objects, sizes, goals, raw).any()
    raw.zero_()
    assert not wait.penalty(roots, velocity, objects, sizes, goals, raw).any()
    wait.reset(torch.tensor([1]), graph, raw)
    penalty = wait.penalty(roots, velocity, objects, sizes, goals, raw)
    assert penalty[1].sum() > 0 and not penalty[[0, 2]].any()


def test_layout_retry_resamples_only_pending_before_scenes_and_keeps_reference_bindings():
    class Task:
        _scene_domains = torch.tensor([0, 1, 2, 0])

        def _reset_actors(self, ids):
            env = ids.repeat_interleave(4)
            owner = torch.arange(4).repeat(len(ids))
            self._reset_ref_slots['loco'] = env, owner
            self._reset_ref_motion_ids['loco'] = 100+4*env+owner
            self._reset_ref_motion_times['loco'] = torch.full_like(env, .5, dtype=torch.float)

    saved = ({'loco': (torch.arange(4).repeat_interleave(4), torch.arange(4).repeat(4))},
             {'loco': torch.arange(16)}, {'loco': torch.zeros(16)})
    merged = resample_before_references(Task(), torch.tensor([0, 1, 2]), saved)
    env, owner = merged[0]['loco']
    assert len(env) == 16 and len(set(zip(env.tolist(), owner.tolist()))) == 16
    fresh = (env == 0) | (env == 2)
    assert torch.equal(merged[1]['loco'], 4*env+owner+100*fresh)
    assert torch.equal(merged[2]['loco'], fresh.float()*.5)
    assert torch.equal(saved[1]['loco'], torch.arange(16))


def test_model_transfer_mixed_forward_gradients_and_reload():
    torch.set_num_threads(1)
    checkpoint = torch.load(native_config('stage2')['experiment']['transfer']['checkpoint'],
                            map_location='cpu', weights_only=False)
    model = torch.nn.Module(); model.a2c_network = build('stage2')
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 26000
    assert all(not p.requires_grad for p in model.a2c_network.actor_encoder.parameters())
    for key, value in checkpoint['model'].items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    net = model.a2c_network
    _, groups, _, _, graph = sampled('mixed', 4)
    obs = _common_observation('solo').expand(4, -1).clone()
    obs[:, -24:-4] = task_graph_packet(graph, 4)
    obs[:, -4:] = groups
    with torch.no_grad():
        for branch in (net.coordination, net.critic_coordination):
            branch.relation_fusion.mlp[-1].weight.normal_(std=.01)
            for block in branch.blocks:
                block.bias_projection.normal_(std=.01)
                block.attention.out_proj.weight.normal_(std=.01)
    mu, _ = net.eval_actor(obs)
    value = net.eval_critic(obs)
    assert mu.shape == (16, 32) and value.shape == (16, 1)
    (mu.square().mean()+value.square().mean()).backward()
    for branch in (net.coordination, net.critic_coordination):
        grad = branch.relation_fusion.embedding.grad
        assert torch.isfinite(grad).all() and (grad.abs().sum(-1) > 0).all()
    restored = build('stage2').eval()
    restored.load_state_dict(net.state_dict(), strict=True)
    for case in ('solo', 'coupled'):
        scene = _common_observation(case)[:, :-4]
        assert torch.equal(goal_aligned_scene(scene, 4, 8), scene)
