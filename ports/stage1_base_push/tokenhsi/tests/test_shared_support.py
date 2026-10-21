from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from utils.shared_support import sample_groups, sample_templates, partition, region_states, capacities, validate_env
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.shared_support_reward import evaluate_shared
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_shared_full_success.yaml').read_text())['env']


def test_configs_differ_only_in_success_geometry():
    other = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage2_shared_region_success.yaml').read_text())['env']
    validate_env(ENV)
    validate_env(other)
    other['sharedSupport']['success_region'] = 'physical'
    assert other == ENV
    assert capacities(ENV) == (12, 18, [2, 2])


@pytest.mark.parametrize('humans,maximum,minimum', [(4,1.6,12), (4,2.4,22), (6,1.6,18), (3,1.6,8), (1,1.6,2)])
def test_token_budget(humans, maximum, minimum):
    env = deepcopy(ENV)
    env['numAgents'] = humans
    env['sharedSupport']['size_ranges'][:2] = [[1.2,maximum]]*2
    env['numObjects'] = minimum
    assert capacities(env)[0] == minimum
    env['numObjects'] = minimum+8
    assert capacities(env)[0] == minimum+8
    env['numObjects'] = minimum-1
    with pytest.raises(ValueError, match=f'at least {minimum}'):
        capacities(env)


@pytest.mark.parametrize('objects', [12,16,24])
def test_packed_tasks_and_real_distractors(objects):
    from utils.shared_support import pack_tokens
    # All shared SIT, two shared ON_TOP pairs, and all solo ON_TOP.
    parents = torch.tensor([[0,1,2,3,12,13,14,15]+[16]*4+[17]*4]*3)
    regions = torch.tensor([[-1]*8+list(range(8))]*3)
    valid = torch.tensor([[False]*8+[True]*4+[False]*4,
                          [True]*4+[False]*4+[True]*8,
                          [True]*8+[False]*8])
    physical = max(16,objects)+2
    mapping, cells, inverse, count = pack_tokens(parents, valid, regions, objects, physical, 4)
    assert count.tolist() == [4,12,8]
    for row in range(3):
        old = valid[row].nonzero().flatten()
        torch.testing.assert_close(mapping[row,inverse[row,old]], parents[row,old])
        torch.testing.assert_close(cells[row,inverse[row,old]], regions[row,old])
        distractors = mapping[row,count[row]:]
        assert len(distractors.unique()) == objects-count[row]
        assert not torch.isin(distractors, parents[row,valid[row]]).any()
        assert not ((distractors >= 16) & (distractors < 18)).any()
        assert (cells[row,count[row]:] == -1).all()
    if objects == 12:
        assert [len(row.unique()) for row in mapping] == [9,6,12]


def test_group_distribution_and_no_joint_carry():
    torch.set_num_threads(1)
    torch.manual_seed(22)
    groups, shared = sample_groups(60000, 4, .8, 'cpu')
    assert abs(shared.float().mean().item()-.8226667) < .005
    assert abs(shared.any(-1).float().mean().item()-.992) < .002
    pair = (groups[:, :, None] == groups[:, None]).sum(-1)
    expected = {1: .008, 2: .1653333, 3: .3466667, 4: .2666667}
    assert abs((pair == 2).all(-1).float().mean().item()-.2133333) < .005
    for maximum, rate in expected.items():
        mask = pair.amax(-1) == maximum
        if maximum == 2:
            mask &= (pair == 1).any(-1)
        assert abs(mask.float().mean().item()-rate) < .005
    tasks = sample_templates(len(groups), 4, ENV['sharedSupport'], shared, 'cpu', 'random_scenario')
    assert not (tasks[shared] == 3).any()
    assert (tasks[shared] == 1).any() and (tasks[shared] == 2).any() and (tasks[shared] == 4).any()
    # A shared group may contain different subtasks.
    same = groups[:, :, None] == groups[:, None]
    assert (same & (tasks[:, :, None] != tasks[:, None])).any()


@pytest.mark.parametrize('m', [1, 2, 3, 4, 6, 8])
def test_groups_cover_all_people(m):
    groups, shared = sample_groups(128, m, .8, 'cpu')
    assert ((groups >= 0) & (groups < m)).all()
    own_leader = groups.gather(1, groups)
    assert torch.equal(own_leader, groups)
    assert torch.equal(shared, (groups[:, :, None] == groups[:, None]).sum(-1) > 1)


def test_partition_coverage_and_rigid_motion():
    size = torch.tensor([[[1.2,1.2,.4], [2.4,2.4,.4]]])
    offsets, cells, valid = partition(size, [3,3])
    assert valid.sum(-1).tolist() == [[4, 9]]
    torch.testing.assert_close((cells[..., 0]*cells[..., 1]*valid).sum(-1), size[..., 0]*size[..., 1])
    assert ((cells[valid][..., :2] >= .4) & (cells[valid][..., :2] <= .800001)).all()
    state = torch.zeros(1,2,13); state[..., 6] = 1
    state[..., :3] = torch.tensor([2.,3.,.2]); state[..., 7] = 1; state[..., 12] = 2
    before = region_states(state, offsets)
    torch.testing.assert_close(before[..., 7], 1-2*offsets[..., 1])
    torch.testing.assert_close(before[..., 8], 2*offsets[..., 0])
    state[..., 0] += 3
    state[..., 5:7] = 2**-.5
    after = region_states(state, offsets)
    torch.testing.assert_close(after[..., 0], 5-offsets[..., 1])
    torch.testing.assert_close(after[..., 1], 3+offsets[..., 0])


def reward_fixture():
    m, o = 4, 12
    graph = compose_canonical_graph(torch.tensor([[1,2,4,3]]), num_objects=o)
    for person in range(3):
        mask = (graph.edge_owner == person) & (graph.edge_relation != 6)
        graph.edge_dst[mask] = m+8
    physical = torch.zeros(1,5,13); physical[..., 6] = 1
    physical[0,4,2] = .2; physical[0,2,2] = .6
    sizes = torch.full((1,5,3), .4); sizes[0,4,:2] = 1.6
    objects = torch.zeros(1,o,13); objects[..., 6] = 1; objects[:,:4] = physical[:,:4]
    obj_sizes = torch.full((1,o,3), .4)
    off, cells, valid = partition(sizes[:,4:], [2,2])
    objects[:,8:] = region_states(physical[:,4:],off).flatten(1,2)
    obj_sizes[:,8:] = cells.flatten(1,2)
    bodies = torch.zeros(1,m,5,3)
    bodies[0,0,0,2] = .52; bodies[0,1,0,2] = 1.4
    bodies[0,1,3:,2] = .4
    task = SimpleNamespace(relation_runtime=SimpleNamespace(graph=graph), num_agents=m, num_objects=o,
        _rigid_body_pos=bodies, _kinematic_humanoid_rigid_body_states=torch.cat((bodies,torch.zeros(1,m,5,10)),-1),
        _key_body_ids=torch.tensor([1,2,3,4]), _box_states=physical, _box_size=sizes,
        _tar_pos=torch.zeros(1,m,3), _relation_cfg=ENV['relationReward'], _char_h=1.,
        _shared=torch.tensor([[True,True,True,False]]), _support_physical=torch.tensor([[4,4,4,-1]]),
        cfg={'env':deepcopy(ENV)})
    task._logical_box_values = lambda values, ids: objects[ids] if values is physical else obj_sizes[ids]
    return task


def test_success_variant_preserves_state_progress_and_saturates_holding():
    task = reward_fixture(); ids = torch.tensor([0])
    phi, full = evaluate_shared(task, ids, True)
    task.cfg['env']['sharedSupport']['success_region'] = 'selected_region'
    region_phi, region = evaluate_shared(task, ids, True)
    torch.testing.assert_close(phi, region_phi)
    torch.testing.assert_close(full['progress'], region['progress'])
    g = task.relation_runtime.graph
    full_success = interaction_own_success(phi, full['z_error'], full['feet_height_error'], g,
                                            task._relation_cfg, full['region_error'])
    region_success = interaction_own_success(phi, region['z_error'], region['feet_height_error'], g,
                                              task._relation_cfg, region['region_error'])
    tasks = g.edge_valid & ((g.edge_relation == 9) | (g.edge_relation == 10) | (g.edge_relation == 8))
    assert full_success[tasks].all() and not region_success[tasks].any()
    runtime = Stage1ContextRuntime(1, g, task._relation_cfg, 'cpu')
    result = runtime.step(phi, full['progress'], full['z_error'], full['feet_height_error'], full['region_error'])
    hold = (g.edge_owner == 2) & (g.edge_relation == 6)
    assert result['reward_saturated'][hold].all()


def test_transfer_padding_mask_and_permutation():
    from learning.multi_agent.shared_support_network import SharedSupportBuilder
    from learning.multi_agent.stage2_network_builder import transfer_stage1
    from torch import nn
    torch.set_num_threads(1)
    train = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage2_shared_full_success.yaml').read_text())
    builder = SharedSupportBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(1420,), amp_input_shape=(1320,),
        value_size=1, num_agents=4, num_objects=12, observation_mode='clean_scene',
        scene_entity_sizes=[223,30,1], scene_kinematic_size=7, scene_arena_scale=16.,
        relation_reward_mode=ENV['relationReward']['mode'], relation_graph_spec=ENV['relationGraph'], device='cpu')
    checkpoint = torch.load(ROOT/'stage1/Stage1V2Fixed_00030500.pth', map_location='cpu', weights_only=False)
    model = nn.Module(); model.a2c_network = net
    report = transfer_stage1(model, checkpoint)
    assert len(report['copied']) == len(checkpoint['model'])
    graph = compose_canonical_graph(torch.tensor([[1,2,3,4]]), num_objects=12)
    nodes = torch.randn(1,4*223+12*30+4)
    poses = torch.zeros(1,20,7); poses[...,6] = 1
    obs = torch.cat((nodes,poses.flatten(1),task_graph_packet(graph,1)),-1)
    valid = torch.zeros(1,12,dtype=torch.bool); valid[:,:8] = True
    encoder = net.actor_encoder
    torch.manual_seed(10); a = encoder(obs,object_valid=valid)
    changed = obs.clone(); changed[:,4*223+8*30:4*223+12*30] += 100
    torch.manual_seed(10); b = encoder(changed,object_valid=valid)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    c = encoder(obs,object_valid=valid,token_order=torch.arange(19,-1,-1))
    torch.testing.assert_close(a,c,rtol=2e-5,atol=2e-5)
    packed = torch.cat((obs,torch.tensor([[17.,17.,-1.,-1.]])),-1)
    mu,_ = net.eval_actor(packed); value = net.eval_critic(packed)
    assert mu.shape == (4,32) and torch.isfinite(value).all()
    (mu.square().sum()+value.square().sum()).backward()
    assert all(p.grad is None for p in net.actor_encoder.parameters())
    assert net.coordination.blocks[0].attention.out_proj.weight.grad.abs().sum() > 0
    from learning.multi_agent.human_coordination import human_relations
    packed[:, -4:] = torch.tensor([[17.,17.,17.,-1.]])
    relation = net._encode_shared(lambda scene: torch.zeros(1,4,64),
        lambda humans, nodes, packet, edge: human_relations(packet, 4), packed)
    assert relation.tolist() == [[[1,2,2,0],[2,1,2,0],[2,2,1,0],[0,0,0,1]]]


def test_checkpoint_allows_larger_evaluation_but_rejects_success_variant():
    from utils.shared_support import checkpoint_contract
    from utils.stage2_spec import check_checkpoint
    weights = {'shared_support_contract': checkpoint_contract(ENV)}
    old = deepcopy(weights)
    old['shared_support_contract']['version'] = 1
    with pytest.raises(ValueError, match='contract differs'):
        check_checkpoint(old, ENV)
    changed = deepcopy(ENV)
    changed['numAgents'] = 8
    changed['sharedSupport']['size_ranges'][:2] = [[2.4,3.2],[2.4,3.2]]
    check_checkpoint(weights, changed)
    changed['sharedSupport']['success_region'] = 'selected_region'
    with pytest.raises(ValueError, match='contract differs'):
        check_checkpoint(weights, changed)
