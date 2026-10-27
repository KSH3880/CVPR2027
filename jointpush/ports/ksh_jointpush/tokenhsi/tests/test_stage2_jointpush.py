import torch

from utils.stage2_jointpush import (sample_groups, group_sizes, sample_templates,
    assign_regions, compose_graph, region_states, group_contact_gate)
from utils.task_role_spec import task_graph_packet


def test_sequential_group_distribution_and_solo_tasks():
    torch.manual_seed(42)
    groups = sample_groups(100000, 4, .8, 'cpu')
    sizes = group_sizes(groups)
    torch.testing.assert_close((sizes > 1).float().mean(), torch.tensor(.8226667), atol=.004, rtol=0)
    torch.testing.assert_close((sizes == 4).all(-1).float().mean(), torch.tensor(4/15), atol=.004, rtol=0)
    tasks = sample_templates(groups, torch.tensor([0., .1, .2, .25, .2, .25]))
    assert (tasks[sizes > 1] == 5).all()
    proportions = torch.bincount(tasks[sizes == 1], minlength=6).float()
    torch.testing.assert_close(proportions/proportions.sum(), torch.tensor([0., .1, .2, .25, .2, .25]), atol=.008, rtol=0)


def test_regions_are_unique_per_group_and_graph_keeps_common_goal():
    groups = torch.tensor([[0,0,2,2], [3,3,3,3], [0,1,1,3]])
    regions = assign_regions(groups)
    for row in range(len(groups)):
        for leader in groups[row].unique():
            assigned = regions[row, groups[row] == leader]
            torch.testing.assert_close(assigned.sort().values, torch.arange(len(assigned)))
    graph = compose_graph(torch.full_like(groups, 5), groups)
    packet = task_graph_packet(graph, len(groups)).reshape(-1,4,5).long()
    torch.testing.assert_close(packet[..., 3], torch.arange(4)[None].expand(3,-1)+4)
    torch.testing.assert_close(packet[..., 4], groups+12)


def test_tokens_follow_translation_rotation_and_angular_velocity():
    boxes = torch.zeros(1, 3, 13)
    boxes[..., 6] = 1
    boxes[..., 12] = 2.
    sizes = torch.tensor([6.,2.,1.5]).expand(1,3,3)
    members = torch.full((1,3),3)
    regions = torch.arange(3)[None]
    states, token_sizes = region_states(boxes, sizes, members, regions)
    torch.testing.assert_close(states[0,:,0], torch.tensor([-2.,0.,2.]))
    torch.testing.assert_close(states[0,:,8], torch.tensor([-4.,0.,4.]))
    torch.testing.assert_close(token_sizes, torch.tensor([2.,2.,1.5]).expand_as(token_sizes))
    boxes[..., 5:7] = 2**-.5
    boxes[..., 0] = 10.
    states, _ = region_states(boxes, sizes, members, regions)
    torch.testing.assert_close(states[0,:,0], torch.full((3,),10.))
    torch.testing.assert_close(states[0,:,1], torch.tensor([-2.,0.,2.]))
    torch.testing.assert_close(states[0,:,7], torch.tensor([4.,0.,-4.]), atol=1e-6, rtol=1e-6)


def test_gate_is_current_group_local_and_not_applied_to_solo():
    boxes = torch.zeros(1,4,13)
    boxes[..., 6] = 1
    sizes = torch.tensor([4.,2.,1.5]).expand(1,4,3)
    goals = torch.tensor([0.,5.,0.]).expand(1,4,3)
    groups = torch.tensor([[0,0,2,3]])
    hands = torch.zeros(1,4,2,3)
    hands[..., 1] = -1.
    assert group_contact_gate(hands, boxes, sizes, goals, groups).all()
    hands[0,1,0,1] = -.81
    assert group_contact_gate(hands, boxes, sizes, goals, groups).tolist() == [[False,False,True,True]]
    hands[0,1,0,1] = -.83
    assert group_contact_gate(hands, boxes, sizes, goals, groups).all()
    assert not group_contact_gate(hands, boxes, sizes, -goals, groups)[0,:2].any()
    assert group_contact_gate(hands, boxes, sizes, boxes[...,:3], groups).all()


def test_bank_covers_every_group_without_aliasing_distinct_objects():
    from pathlib import Path
    import yaml
    from utils.stage2_jointpush import asset_bank, physical_assignment, validate_env
    root = Path(__file__).resolve().parents[2]
    env = yaml.safe_load((root/'tokenhsi/data/cfg/multi_agent/stage2_jointpush.yaml').read_text())['env']
    validate_env(env)
    sizes, masses = asset_bank(512,env,'cpu')
    torch.testing.assert_close(sizes[:,20:22,0],2*sizes[:,16:18,0])
    torch.testing.assert_close(sizes[:,22:24,0],sizes[:,16:17,0]*torch.tensor([3,4]))
    torch.testing.assert_close(masses[:,16:],torch.tensor([20.,20.,20.,20.,40.,40.,60.,80.]).expand(512,-1))
    groups = sample_groups(512,4,.8,'cpu')
    templates = sample_templates(groups,torch.tensor([0.,.1,.2,.25,.2,.25]))
    physical = physical_assignment(templates,groups)
    torch.testing.assert_close(physical[:,:,None] == physical[:,None],groups[:,:,None] == groups[:,None])
    assert physical.min() >= 0 and physical.max() < 24


def test_gate_blocks_all_shared_at_components_but_keeps_contact_and_solo():
    from pathlib import Path
    import yaml
    from env.tasks.multi_agent.stage2_jointpush_reward import Stage2JointPushRuntime
    env = yaml.safe_load((Path(__file__).resolve().parents[2]/'tokenhsi/data/cfg/multi_agent/stage2_jointpush.yaml').read_text())['env']
    graph = compose_graph(torch.full((1,4),5),torch.tensor([[0,0,2,3]]),False)
    runtime = Stage2JointPushRuntime(1,graph,env['relationReward'],'cpu',contact_weights=[.2]*3)
    runtime.gate[0,:2] = False
    runtime.contact[:] = True
    one,zero = torch.ones(1,8),torch.zeros(1,8)
    result = runtime.step(one,one,zero,zero,zero)
    shared_at = (graph.edge_relation == 7)&(graph.edge_owner < 2)
    assert (result['total'][shared_at] == 0).all()
    torch.testing.assert_close(result['agent_task_reward'],torch.tensor([[.6,.6,1.2,1.2]]))
    runtime.gate[:] = True
    runtime.contact[:] = False
    result = runtime.step(one,one,zero,zero,zero)
    torch.testing.assert_close(result['agent_task_reward'],torch.full((1,4),1.2))


def test_swept_rectangle_rejects_side_obstacles_and_crossing_paths():
    from utils.stage2_jointpush_scene import swept_overlap
    start,end = torch.tensor([0.,0.]),torch.tensor([0.,8.])
    yaw,half = torch.tensor(0.),torch.tensor([4.,1.])
    obstacle = torch.tensor([[3.8,4.],[4.7,4.],[0.,10.]])
    hit = swept_overlap(start,end,yaw,half,obstacle,obstacle,torch.zeros(3),torch.full((3,2),.25),.1)
    assert hit.tolist() == [True,False,False]
    assert swept_overlap(start,end,yaw,half,torch.tensor([-7.,4.]),torch.tensor([7.,4.]),yaw,half)
    assert not swept_overlap(start,end,yaw,half,torch.tensor([-7.,12.]),torch.tensor([7.,12.]),yaw,half)


def test_stage1_transfer_and_identity_coordination():
    from pathlib import Path
    import yaml
    from torch import nn
    from learning.multi_agent.stage2_jointpush_network import Stage2JointPushBuilder, transfer_stage1
    root = Path(__file__).resolve().parents[2]
    checkpoint = root/'pth/Stage1PushFinetune_00010500.pth'
    if not checkpoint.exists():
        import pytest
        pytest.skip('User checkpoint is not distributed with the repository')
    env = yaml.safe_load((root/'tokenhsi/data/cfg/multi_agent/stage2_jointpush.yaml').read_text())['env']
    params = yaml.safe_load((root/'tokenhsi/data/cfg/train/rlg/stage1_jointpush.yaml').read_text())['params']['network']
    builder = Stage2JointPushBuilder()
    builder.load(params)
    model = nn.Module()
    model.a2c_network = builder.build('amp', actions_num=32, input_shape=(1272,), amp_input_shape=(1330,),
        value_size=1, num_agents=4, num_objects=8, observation_mode='clean_scene', scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7, scene_arena_scale=16., relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'], device='cpu')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    report = transfer_stage1(model, weights)
    assert len(report['copied']) == 165 and len(report['added']) > 0
    net = model.a2c_network
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    assert all(p.requires_grad for p in net.coordination.parameters())
    obs = torch.zeros(1,1272)
    obs[:,1136:1248].reshape(1,16,7)[...,6] = 1.
    groups = torch.tensor([[0,0,2,3]])
    graph = compose_graph(torch.tensor([[5,5,1,4]]), groups)
    obs[:,1248:1268] = task_graph_packet(graph,1)
    obs[:,-4:] = groups
    net.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        reference = net.action_head(net.actor_encoder(obs[:,:-4])).reshape(4,-1)
        torch.manual_seed(7)
        action, _ = net.eval_actor(obs)
        value = net.eval_critic(obs)
    torch.testing.assert_close(action, reference)
    assert action.shape == (4,32) and value.shape == (4,1)
    assert torch.isfinite(action).all() and torch.isfinite(value).all()
