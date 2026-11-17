from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from utils.experiments import native_config, validate_experiment
from utils import stage2_spec
from utils.sensor_stage2_spec import bridge_roles, sample_surfaces, BRIDGE
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder, transfer
from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene


def build(builder_class, name, width):
    env = native_config(name)['env']
    builder = builder_class()
    builder.load(native_config(name,True)['params']['network'])
    net = builder.build('amp',actions_num=32,input_shape=(width,),amp_input_shape=(1330,),value_size=1,
        num_agents=4,num_objects=8,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7,scene_arena_scale=16.,relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=deepcopy(env['relationGraph']),device='cpu')
    for encoder in (net.actor_encoder,net.critic_encoder):
        encoder.relation_graph_spec['shuffle_token_order'] = False
    net.coordination.shuffle_human_order = net.critic_coordination.shuffle_human_order = False
    model=nn.Module();model.a2c_network=net
    return model.eval()


@pytest.mark.parametrize('experiment', ['stage2_sensor_900', 'stage2_sensor_900_warehouse', 'stage2_sensor_900_warehouse_bridge_init', 'stage2_sensor_900_warehouse_bridge_init_rsi', 'stage2_sensor_900_room_warehouse'])
def test_sensor_initial_equivalence_mask_and_frozen_updates(experiment):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    source=build(Stage2PushBuilder,'stage2_distill_900',1272)
    target=build(SensorStage2Builder,experiment,10488)
    transfer(target,dict(model=source.state_dict(),relation_experiment_config=native_config('stage2_distill_900')))
    env=native_config('stage2_distill_900')['env']
    _,groups,pre,tasks=stage2_spec.sample_episode(torch.full((4,4,3),.4),env['box']['build']['taskSizeRanges'],env['relationGraph']['template_probabilities'])
    graph=stage2_spec.compose_graph(tasks,groups,pre,8)
    poses=torch.zeros(4,16,7);poses[...,:3]=torch.randn(4,16,3);poses[...,6]=1
    obs=torch.cat((torch.randn(4,1136),poses.flatten(1),task_graph_packet(graph,4).float(),groups.float()),-1)
    full=torch.cat((obs,torch.randn(4,9216)),-1)
    original={k:v.clone() for k,v in source.state_dict().items()}
    net=target.a2c_network
    for method in ('eval_actor','eval_critic'):
        actual=getattr(net,method)(full);expected=getattr(source.a2c_network,method)(obs)
        if isinstance(actual,tuple):actual,expected=actual[0],expected[0]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    optimizer=torch.optim.Adam([p for p in target.parameters() if p.requires_grad],lr=2e-5)
    for _ in range(2):
        loss=net.eval_actor(full)[0].square().mean()+net.eval_critic(full).square().mean()
        loss=loss+net.eval_disc(torch.randn(4,1330)).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    for name,value in original.items():
        if name.startswith(('a2c_network.actor_encoder.', 'a2c_network.coordination.')) or name=='a2c_network.sigma':
            assert torch.equal(value,target.state_dict()[name]),name
    for prefix in ('critic_encoder.', 'critic_coordination.', 'action_head.', 'value_head.', '_disc_mlp.'):
        keys=[k for k in original if k.startswith('a2c_network.'+prefix)]
        assert any(not torch.equal(original[k],target.state_dict()[k]) for k in keys),prefix
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    assert all(not p.requires_grad for p in net.coordination.parameters())
    assert all(p.requires_grad for p in net.critic_encoder.parameters())
    assert all(p.requires_grad for p in net.critic_coordination.parameters())
    assert net.actor_sensor[0].weight.grad.abs().sum()>0
    assert net.critic_sensor[0].weight.grad.abs().sum()>0
    net.mask_sensor=True
    torch.testing.assert_close(net.eval_actor(full)[0],Stage2PushBuilder.Network.eval_actor(net,obs)[0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(net.eval_critic(full),Stage2PushBuilder.Network.eval_critic(net,obs),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('count',[1,2,3])
def test_bridge_shared_roles_sizes_and_gap(count):
    env=native_config('stage2_sensor_900')['env'];n=512
    sizes=torch.empty(n,4,3).uniform_(.4,.5)
    groups,pre,tasks,roles,push=bridge_roles(sizes,env['box']['build']['taskSizeRanges'],env['relationGraph']['template_probabilities'],torch.full((n,),count))
    assert (push.sum(-1)==count).all()
    assert not push[torch.arange(n),roles[:,1]].any()
    assert (pre[torch.arange(n),roles[:,1]]==-1).all()
    graph=stage2_spec.compose_graph(tasks,groups,pre,8)
    mapping,primary,_=stage2_spec.physical_mapping(tasks,groups,pre,graph)
    leader=primary.gather(1,roles[:,:1])
    assert ((primary==leader)|~push).all()
    if count==1:
        assert .4<float((pre>=0).any(-1).float().mean())<.6
    else:
        assert (pre==-1).all()
    bridge=torch.tensor([[1.2*count,.8,.5]])
    _,gap,_=scene.bridge_stages(torch.full((1,2),5.),torch.full((1,2),5.),bridge,env['sensorScene']['bridge'])
    torch.testing.assert_close(gap,bridge[:,0]+.2)
    surface,start,unreachable=sample_surfaces(tasks,pre,roles,torch.full((n,),BRIDGE),push)
    assert (start[push]==0).all()
    assert (surface[torch.arange(n),roles[:,1]]!=start[torch.arange(n),roles[:,1]]).all()
    assert not unreachable.any()


def test_sensor_config_and_leader_common_intersection():
    env=native_config('stage2_sensor_900')['env']
    validate_experiment('stage2_sensor_900',env,native_config('stage2_sensor_900',True))
    from utils.before_spec import eligible_families
    grid=torch.cartesian_prod(torch.arange(8,13),torch.arange(8,13),torch.arange(7,11)).float()/20
    assert eligible_families(grid[None],env['box']['build']['taskSizeRanges']).all()
    assert native_config('stage2_distill_900',True)['params']['algo']['name']=='ma_distill'


def test_grid_is_root_centred_and_follows_heading():
    grid = sensor.sensor_grid(48, 7., 'cpu')
    assert grid.shape == (2304, 2) and grid.abs().max() == 7.
    root = torch.tensor([[2., 3., .9]])
    points = sensor.sensor_positions(root, torch.tensor([math.pi/2]), grid)
    assert torch.allclose(points.mean(1), root[:, :2], atol=1e-5)
    # The grid corner (+7, 0 side) rotates onto +y with the heading.
    index = grid[:, 0].argmax()+24
    forward = points[0, index]-root[0, :2]
    assert forward[1] > 6. and forward[0].abs() < .3


def test_vertical_height_and_observation():
    centers = torch.tensor([[[0., 0., .75], [0., 0., scene.HIDDEN]]])
    quats = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])
    half = torch.tensor([[[2., 2., .75], [0., 0., 0.]]])
    xy = torch.tensor([[[0., 0.], [1.9, 0.], [2.1, 0.], [9., 9.]]])
    height = sensor.vertical_height(xy, centers, quats, half, torch.zeros(1))
    assert torch.allclose(height, torch.tensor([[1.5, 1.5, 0., 0.]]))
    # Yawed footprint: a corner of the unrotated square falls outside after 45 degrees.
    yawed = scene.yaw_quat(torch.tensor([[math.pi/4, 0.]]))
    height = sensor.vertical_height(torch.tensor([[[1.9, 1.9], [2.7, 0.]]]), centers, yawed, half, torch.zeros(1))
    assert torch.allclose(height, torch.tensor([[0., 1.5]]))
    roots = torch.tensor([[[0., 0., 2.4], [20., 0., .9]]])
    grid = sensor.sensor_grid(3, 1., 'cpu')
    obs = sensor.height_observation(roots, torch.zeros(1, 2), grid, centers, quats, half, torch.zeros(1), 3.)
    assert obs.shape == (1, 2, 9)
    assert torch.allclose(obs[0, 0], torch.full((9,), -.3), atol=1e-5)
    assert torch.allclose(obs[0, 1], torch.full((9,), -.3), atol=1e-5)
    assert sensor.height_observation(roots+torch.tensor([0., 0., 9.]), torch.zeros(1, 2), grid, centers, quats,
                                     half, torch.zeros(1), 3.).min() == -1.


def test_rigid_placement_preserves_group_geometry():
    torch.manual_seed(4)
    n = 512
    xy = torch.rand(n, 4, 2)*6-3
    margin = torch.full((n, 4), .8)
    valid = torch.ones(n, 4, dtype=torch.bool)
    center = torch.tensor([3., -2.]).expand(n, 2)
    half = torch.full((n, 2), 6.)
    angle, offset, placed = scene.place_rigid(xy, margin, valid, center, half, lambda *_: True, 16)
    assert placed.all()
    final = scene.rotate_xy(xy, angle[:, None])+offset[:, None]
    assert ((final-center[:, None]).abs() <= half[:, None]-.8+1e-4).all()
    before = (xy[:, :, None]-xy[:, None]).norm(dim=-1)
    after = (final[:, :, None]-final[:, None]).norm(dim=-1)
    assert torch.allclose(before, after, atol=1e-4)
    # A group wider than the surface can never be accepted.
    wide = torch.tensor([[[-9., 0.], [9., 0.]]]).repeat(4, 1, 1)
    _, _, fitted = scene.place_rigid(wide, torch.full((4, 2), .8), torch.ones(4, 2, dtype=torch.bool),
                                     torch.zeros(4, 2), torch.full((4, 2), 5.), lambda *_: True, 16)
    assert not fitted.any()


def test_offstage_layout_is_out_of_reach():
    torch.manual_seed(5)
    n = 4096
    half = torch.rand(n, 2)*2+5
    radius = torch.rand(n)*.3+.2
    second = torch.rand(n)*4+1
    human, box, goal = scene.offstage_layout(half, radius, second, [.8, 1.5], .8, 5.)
    assert ((human.abs() <= half-.8+1e-4).all(-1)).all()
    outside = ((box.abs()-half).amax(-1))
    assert (outside >= radius+.8-1e-4).all() and (outside <= radius+1.5+1e-4).all()
    assert ((goal.abs()-half).amax(-1) >= radius+.8-1e-4).all()
    distance = (human-box).norm(dim=-1)
    assert (distance >= 1.-1e-4).all() and (distance <= 5.+1e-4).all()
    assert torch.allclose((goal-box).norm(dim=-1), second, atol=1e-4)


def test_bridge_layout_and_crosser_alignment():
    torch.manual_seed(6)
    env = native_config('stage2_sensor_900')['env']
    cfg = env['sensorScene']['bridge']
    n = 2048
    half_a, half_b = torch.rand(n, 2)*2+4, torch.rand(n, 2)*2+4
    bridge = torch.tensor([1.8, 1.9, 1.5]).expand(n, 3)
    far, gap, bounds = scene.bridge_stages(half_a, half_b, bridge, cfg)
    assert torch.allclose(gap, torch.full((n,), 2.)) and (far[:, 0] == half_a[:, 0]+gap+half_b[:, 0]).all()
    near, side = torch.rand(n) < .5, torch.rand(n) < .5
    layout = scene.bridge_layout(half_a, half_b, far, gap, bridge, cfg, near, side)
    landing = layout['landing']
    need = 1.9/2+cfg['landing_margin']
    assert (landing[:, 1] >= torch.maximum(-half_a[:, 1], far[:, 1]-half_b[:, 1])+need-1e-4).all()
    assert (landing[:, 1] <= torch.minimum(half_a[:, 1], far[:, 1]+half_b[:, 1])-need+1e-4).all()
    # The box starts on the channel axis and the pusher stands behind it, facing the landing.
    assert (layout['start'][:, 0] == landing[:, 0]).all()
    assert torch.allclose((layout['pusher']-layout['start'])[:, 1], layout['sign']*cfg['pusher_stand_off'], atol=1e-4)
    assert (((landing-layout['start'])[:, 1]*layout['sign']) <= 0).all()
    assert (layout['pusher'][:, 1] >= bounds[:, 2]).all() and (layout['pusher'][:, 1] <= bounds[:, 3]).all()
    task = torch.randint(1, 5, (n,))
    aligned = torch.rand(n) < .5
    centers_a, centers_b = torch.zeros(n, 2), far
    r = torch.full((n,), .4)
    box, second, human, failed = scene.crosser_layout(task, centers_b, half_b, centers_a, half_a, r, r, landing, bridge,
                                                      gap, aligned, .5, .8, cfg)
    assert not failed.any()
    assert ((box-far).abs() <= half_b-.9+1e-4).all()
    assert ((human.abs() <= half_a-.8+1e-3).all(-1)).all()
    # Aligned starts see the box straight through the landing point.
    a = aligned.nonzero().flatten()
    cross = ((human-landing)[a, 0]*(box-landing)[a, 1]-(human-landing)[a, 1]*(box-landing)[a, 0]).abs()
    assert (cross < 1e-2).all()
    assert ((box-landing).norm(dim=-1) >= gap/2+cfg['landing_box_clearance']+r-1e-4).all()


def test_bridge_carry_pickup_on_both_stages():
    torch.manual_seed(16)
    cfg = native_config('stage2_sensor_900')['env']['sensorScene']['bridge']
    task = torch.tensor([1, 2, 3, 4, 3, 4]).repeat(512)
    n = len(task)
    carry_from_start = torch.tensor([False, False, False, False, True, True]).repeat(512)
    half_a, half_b = torch.rand(n, 2)*2+4, torch.rand(n, 2)*2+4
    bridge = torch.tensor([1.8, 1.9, 1.5]).expand(n, 3)
    far, gap, _ = scene.bridge_stages(half_a, half_b, bridge, cfg)
    landing = scene.bridge_layout(half_a, half_b, far, gap, bridge, cfg,
        torch.rand(n) < .5, torch.rand(n) < .5)['landing']
    reverse = torch.rand(n) < .5
    start = torch.where(reverse[:, None], far, torch.zeros_like(far))
    dest = torch.where(reverse[:, None], torch.zeros_like(far), far)
    start_half = torch.where(reverse[:, None], half_b, half_a)
    dest_half = torch.where(reverse[:, None], half_a, half_b)
    radius = torch.full((n,), .4)
    aligned = torch.rand(n) < .5
    box, second, human, failed = scene.crosser_layout(task, dest, dest_half, start, start_half,
        radius, radius, landing, bridge, gap, aligned, .5, .8, cfg, carry_from_start)
    assert not failed.any()
    expected = torch.where(carry_from_start[:, None], start, dest)
    half = torch.where(carry_from_start[:, None], start_half, dest_half)
    assert ((box-expected).abs() <= half-.9+1e-4).all()
    assert ((human-start).abs() <= start_half-.8+1e-4).all()
    carry = (task == 3) | (task == 4)
    assert ((second-dest).abs()[carry] <= dest_half[carry]-.9+1e-4).all()
    assert ((human-box).norm(dim=-1)[carry_from_start] >= 1.05-1e-4).all()
    across = torch.where(carry_from_start[:, None], second, box)-landing
    for points, chosen in ((human, aligned), (box, aligned & carry_from_start)):
        offset = points-landing
        cross = offset[:, 0]*across[:, 1]-offset[:, 1]*across[:, 0]
        assert (cross[chosen].abs() < 1e-2).all()
    chosen = aligned & carry_from_start
    assert ((human[:, 0]-box[:, 0])*(box[:, 0]-landing[:, 0]))[chosen].gt(0).all()


def test_room_walls_and_wall_cpa():
    half = torch.tensor([[6., 5.]])
    centers, sizes, starts, ends = scene.room_walls(half, .4, 2.5, 1.8)
    assert centers.shape == (1, 5, 3) and starts.shape == (1, 8, 2)
    assert torch.allclose(sizes[0, 4], torch.tensor([.2, 3.2, 1.25]))
    assert torch.allclose(centers[0, 0, :2], torch.tensor([6.2, 0.]))
    wall_start, wall_end = torch.tensor([[[1., -5.]]]), torch.tensor([[[1., 5.]]])
    root = torch.tensor([[[0., 0., .9]]])
    dt = 1/30
    toward = scene.compute_wall_cpa_collision_penalty(root, torch.tensor([[[1., 0., 0.]]]), wall_start, wall_end, .7, .95, dt)
    assert toward.item() == pytest.approx(.95**30, abs=1e-4)
    away = scene.compute_wall_cpa_collision_penalty(root, torch.tensor([[[-1., 0., 0.]]]), wall_start, wall_end, .7, .95, dt)
    assert away.item() == 0.
    along = scene.compute_wall_cpa_collision_penalty(root, torch.tensor([[[0., 1., 0.]]]), wall_start, wall_end, .7, .95, dt)
    assert along.item() == 0.


def test_terrain_fall_by_start_surface():
    contact = torch.zeros(2, 1, 3, 3)
    contact[:, :, 0, 2] = 1.  # a foot touches something
    body = torch.tensor([[[[0., 0., .05], [0., 0., .9], [0., 0., 1.5]]]]).repeat(2, 1, 1, 1)
    thresholds = torch.tensor([.1, .15, .3])
    floor = torch.zeros(2, 1, 3)
    progress = torch.full((2,), 5)
    start_floor = torch.tensor([[True], [False]])
    fallen = scene.terrain_fall(contact, body, torch.tensor([0]), floor, thresholds, progress, torch.zeros(2), start_floor)
    assert fallen.tolist() == [[False], [True]]
    # The same foot on a raised surface is not a floor contact.
    raised = torch.full((2, 1, 3), 1.)
    body[..., 2] += 1.
    fallen = scene.terrain_fall(contact, body, torch.tensor([0]), raised, thresholds, progress, torch.zeros(2), start_floor)
    assert not fallen.any()
    assert not scene.terrain_fall(contact, body*0, torch.tensor([0]), floor, thresholds, torch.zeros(2), torch.zeros(2), start_floor).any()

