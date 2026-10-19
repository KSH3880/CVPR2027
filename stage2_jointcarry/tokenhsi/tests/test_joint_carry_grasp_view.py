from copy import deepcopy

import pytest
import torch

from learning.multi_agent.grasp_view import GRASP_VIEW, make_grasp_views, encode_grasp_views
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.stage1_grasp_shift import shared_payload_views, initial_grasp_sides
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry_amp_ablation import configs, network
from test_joint_carry import build_network
from test_stage2_coordination import Wrapper
from test_stage1_grasp_shift import scene

BASE = 'locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding'
NEW = BASE.replace('_task_embedding', '_graspview_task_embedding')


def observations():
    obs, _, end = scene()
    obs = obs.expand(4, -1).clone()
    obs[:2, end:] = torch.tensor([1, 2, 0, 2, 6, 1, 2, 1, 2, 6])
    obs[1, end:] = torch.tensor([1, 3, 0, 2, 4, 1, 3, 1, 2, 4])
    return obs


def test_config_only_adds_observation_contract_and_parameters_match():
    env, train = configs(NEW)
    base, old = configs(BASE)
    validate_env(env)
    validate_ablation(env, train)
    assert env.pop('jointCarryGraspView') == GRASP_VIEW
    assert env == base
    assert train['params']['network']['coordination'].pop('grasp_view') is True
    assert train['params']['network'] == old['params']['network']
    a, b = network(BASE), network(NEW)
    assert {k:v.shape for k,v in a.state_dict().items()} == {k:v.shape for k,v in b.state_dict().items()}
    with pytest.raises(ValueError, match='grasp-view'):
        check_ablation_checkpoint({}, dict(base, jointCarryGraspView=GRASP_VIEW))
    with pytest.raises(ValueError, match='grasp-view'):
        check_ablation_checkpoint({'joint_carry_grasp_view': GRASP_VIEW}, base)


def test_views_match_stage1_toy_solo_unchanged_and_packet_shuffle():
    obs = observations()
    original = obs.clone()
    signs = torch.tensor([[-1., 1.]]).expand(4, -1)
    views = make_grasp_views(obs, signs, .15).reshape(4, 2, -1)
    torch.testing.assert_close(views[:2].flatten(0, 1), shared_payload_views(obs[:2], signs[:2], 1.))
    torch.testing.assert_close(views[2:], obs[2:, None].expand(-1, 2, -1), atol=0, rtol=0)
    shuffled = obs.clone()
    shuffled[:, -10:] = obs[:, -10:].reshape(4, 2, 5).flip(1).flatten(1)
    actual = make_grasp_views(shuffled, signs, .15).reshape(4, 2, -1)
    torch.testing.assert_close(actual[..., :-10], views[..., :-10], atol=0, rtol=0)
    torch.testing.assert_close(obs, original, atol=0, rtol=0)


def test_normalizer_preserves_source_state_and_normalizes_each_view():
    kwargs = dict(entity_sizes=[223,30,1], entity_counts=[2,4,2],
                  normalized_sizes=[223,30,0], kinematic_size=7, extra_passthrough_size=10)
    single = SceneRunningMeanStd(**kwargs).eval()
    dual = SceneRunningMeanStd(**kwargs, num_views=2).eval()
    for rms in single.running_mean_std:
        rms.running_mean.normal_()
        rms.running_var.uniform_(.2, 2.)
    dual.load_state_dict(single.state_dict(), strict=True)
    views = make_grasp_views(observations(), torch.tensor([[-1.,1.]]).expand(4,-1), .15)
    expected = single(views.reshape(-1, 634)).reshape(4, -1)
    torch.testing.assert_close(dual(views), expected, atol=0, rtol=0)
    torch.testing.assert_close(dual(views).reshape(4,2,-1)[...,568:], views.reshape(4,2,-1)[...,568:])


def test_transfer_matches_per_agent_stage1_and_critic_gradients():
    torch.set_num_threads(1)
    source = Wrapper(build_network(False))
    target = network(NEW)
    # Copy a schema-9 semantic source, as in the existing transfer tests.
    import yaml
    from test_joint_carry_amp_ablation import ROOT
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    transfer_stage1_weights(target, dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward)))
    views = make_grasp_views(observations(), torch.tensor([[-1.,1.]]).expand(4,-1), .15)
    net = target.a2c_network
    with torch.no_grad():
        all_actions = source.a2c_network.eval_actor(views.reshape(-1,634))[0].reshape(4,2,2,32)
        expected = torch.stack((all_actions[:,0,0],all_actions[:,1,1]),1).flatten(0,1)
        torch.testing.assert_close(net.eval_actor(views)[0], expected, atol=2e-6, rtol=1e-5)
        all_values = source.a2c_network.eval_critic(views.reshape(-1,634)).reshape(4,2,2,1)
        expected_value = torch.stack((all_values[:,0,0],all_values[:,1,1]),1).flatten(0,1)
        torch.testing.assert_close(net.eval_critic(views), expected_value, atol=2e-6, rtol=1e-5)
    counts = []
    handle = net.critic_encoder.register_forward_pre_hook(lambda module, args: counts.append(len(args[0])))
    net.eval_critic(views).square().mean().backward()
    handle.remove()
    assert counts == [6]  # Two joint scenes twice, two solo scenes once.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.critic_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in net.actor_encoder.parameters())


def test_encoder_selection_matches_unoptimized_forward_and_gradients():
    net = network(NEW).a2c_network
    obs = make_grasp_views(observations(), torch.tensor([[-1.,1.]]).expand(4,-1), .15)
    oracle = deepcopy(net.critic_encoder)
    actual = encode_grasp_views(net.critic_encoder, obs)
    all_humans = oracle(obs.reshape(-1,634)).reshape(4,2,2,64)
    expected = torch.stack((all_humans[:,0,0],all_humans[:,1,1]),1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    actual.square().mean().backward(); expected.square().mean().backward()
    for a,b in zip(net.critic_encoder.parameters(),oracle.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)


def test_human_swap_keeps_anchor_roles_and_output_permutation():
    obs = observations()
    poses = obs[:,568:624].reshape(4,8,7)
    poses[:,0,1], poses[:,1,1] = -2., 2.
    swapped = obs.clone()
    swapped[:,:446] = obs[:,:446].reshape(4,2,223).flip(1).flatten(1)
    swapped[:,566:568] = obs[:,566:568].flip(1)
    swapped[:,568:624] = poses[:,[1,0,2,3,4,5,7,6]].flatten(1)
    packet = obs[:,-10:].reshape(4,2,5).flip(1).clone()
    packet[...,2] = 1 - packet[...,2]
    packet[...,4] = torch.where(packet[...,4]>=6,13-packet[...,4],packet[...,4])
    swapped[:,-10:] = packet.flatten(1)
    torch.testing.assert_close(initial_grasp_sides(swapped),initial_grasp_sides(obs).flip(1))
    net = network(NEW).a2c_network
    for module in (net.coordination,net.critic_coordination):
        for block in module.blocks:
            torch.nn.init.normal_(block.attention.out_proj.weight,std=.02)
            torch.nn.init.normal_(block.ffn[-1].weight,std=.02)
    a = make_grasp_views(obs,initial_grasp_sides(obs),.15)
    b = make_grasp_views(swapped,initial_grasp_sides(swapped),.15)
    with torch.no_grad():
        torch.testing.assert_close(net.eval_actor(a)[0].reshape(4,2,-1).flip(1),
            net.eval_actor(b)[0].reshape(4,2,-1),atol=3e-6,rtol=1e-5)
        torch.testing.assert_close(net.eval_critic(a).reshape(4,2,-1).flip(1),
            net.eval_critic(b).reshape(4,2,-1),atol=3e-6,rtol=1e-5)


def six_observations():
    from learning.multi_agent.grasp_view import scene_parts
    obs = torch.randn(3, 243*6+37*8) * .1
    objects, poses = scene_parts(obs, 6)
    poses[..., 3:] = torch.tensor([0., 0., 0., 1.])
    poses[:, :6, 1] = torch.tensor([-2., 2., 1., -1., 3., 4.])
    objects[..., 6:] = torch.tensor([-.26, -.4, -.2, .26, .4, .2]*4)
    obs[:, -30:] = torch.tensor([1,2,0,6,14, 1,2,1,6,14, 1,2,2,7,16,
                                1,2,3,7,16, 1,2,4,8,18, 1,3,5,9,10])
    return obs


def test_six_views_shift_only_own_coupled_payload_and_packet_shuffle():
    from learning.multi_agent.grasp_view import initial_grasp_sides as assign, scene_parts
    obs = six_observations()
    sides = assign(obs, 6)
    torch.testing.assert_close(sides, torch.tensor([[-1.,1.,1.,-1.,0.,0.]]).expand(3,-1))
    views = make_grasp_views(obs, sides, .15).reshape(3,6,-1)
    torch.testing.assert_close(views[:,4:], obs[:,None].expand(-1,2,-1), atol=0, rtol=0)
    expected = obs[:,None].expand(-1,6,-1).clone()
    boxes, poses = scene_parts(expected, 6)
    for owner, payload in enumerate([0,0,1,1]):
        delta = torch.zeros(3,3); delta[:,1] = .25*sides[:,owner]
        poses[:,owner,6+payload,:3] += delta
        boxes[:,owner,payload,:3] += torch.cross(boxes[:,owner,payload,3:6],delta,dim=-1)
    torch.testing.assert_close(views, expected)
    shuffled = obs.clone()
    shuffled[:,-30:] = obs[:,-30:].reshape(3,6,5)[:,[5,3,1,4,2,0]].flatten(1)
    torch.testing.assert_close(assign(shuffled,6), sides)
    torch.testing.assert_close(make_grasp_views(shuffled,sides,.15).reshape(3,6,-1)[...,:-30],views[...,:-30])


def test_six_encoder_selection_and_human_permutation():
    from learning.multi_agent.grasp_view import initial_grasp_sides as assign, scene_parts
    obs = six_observations()
    net = network(NEW).a2c_network
    from utils.joint_carry_demo import demo_graph_spec
    for encoder in (net.actor_encoder, net.critic_encoder):
        encoder.relation_graph_spec = demo_graph_spec()
    net.set_entity_counts(6,8)
    net.eval()
    for coord in (net.coordination,net.critic_coordination):
        for block in coord.blocks:
            torch.nn.init.normal_(block.attention.out_proj.weight,std=.02)
    views = make_grasp_views(obs,assign(obs,6),.15)
    with torch.no_grad():
        full = net.actor_encoder(views.reshape(-1,obs.shape[-1])).reshape(3,6,6,64)
        expected = full[:,torch.arange(6),torch.arange(6)]
        torch.testing.assert_close(encode_grasp_views(net.actor_encoder,views),expected,atol=3e-6,rtol=1e-5)
    order = torch.tensor([4,2,0,5,1,3]); inverse = order.argsort()
    swapped = obs.clone()
    swapped[:,:1338] = obs[:,:1338].reshape(3,6,223)[:,order].flatten(1)
    swapped[:,1578:1584] = obs[:,1578:1584][:,order]
    _, poses = scene_parts(obs,6); _, new_poses = scene_parts(swapped,6)
    new_poses[:] = poses[:,torch.cat((order,torch.arange(6,14),order+14))]
    packet = swapped[:,-30:].reshape(3,6,5)
    packet[...,2] = inverse[packet[...,2].long()]
    packet[...,4] = torch.where(packet[...,4]>=14,inverse[(packet[...,4].long()-14).clamp(0,5)]+14,packet[...,4])
    sides = assign(swapped,6)
    torch.testing.assert_close(sides,assign(obs,6)[:,order])
    moved = make_grasp_views(swapped,sides,.15)
    with torch.no_grad():
        torch.testing.assert_close(net.eval_actor(views)[0].reshape(3,6,-1)[:,order],net.eval_actor(moved)[0].reshape(3,6,-1),atol=4e-6,rtol=1e-5)
        torch.testing.assert_close(net.eval_critic(views).reshape(3,6,-1)[:,order],net.eval_critic(moved).reshape(3,6,-1),atol=4e-6,rtol=1e-5)
