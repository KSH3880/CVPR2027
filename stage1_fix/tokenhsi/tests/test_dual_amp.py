"""TeamHOI semantics and scene/replay integrity; no simulator creation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi  # Required before torch; no simulator is created.
from types import SimpleNamespace
import copy
import pytest
import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.replay_buffer import ReplayBuffer
from learning.multi_agent.dual_amp_agent import DualAMPAgent
from learning.multi_agent.dual_amp_model import DualAMPBuilder,DualAMPModel
from utils.dual_amp import (mask_history,blend_weight,outside_history,blend_rewards,
                           style_reward,rim_distance,check_dual_checkpoint)
from utils.lift_spec import ARM_FEATURES

torch.set_num_threads(1)


def make_agent():
    train=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1_teamhoi.yaml').read_text())
    params=train['params']['network']
    params['disc']['units']=[32,16]
    params['transformer'].update(num_layers=1,layer_dim_feedforward=64,extra_mlp_units=[32],tokenizer_units=[32])
    builder=DualAMPBuilder();builder.load(params)
    net=builder.build('amp',actions_num=32,input_shape=(550,),amp_input_shape=(1290,),
        value_size=1,num_agents=2,num_objects=2,humanoid_obs_size=230,object_obs_size=39,
        goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7,scene_arena_scale=5.,relation_graph_spec={'template':'support_lift_v1'},device='cpu')
    for encoder in (net.actor_encoder,net.critic_encoder):encoder.gta_diagnostics_first_forward=False
    agent=object.__new__(DualAMPAgent)
    agent._amp_input_mean_std=RunningMeanStd((1290,))
    agent._amp_masked_input_mean_std=RunningMeanStd((1290,))
    agent._dual_cfg={'distance_threshold':1.,'sharpness':15.,'full_min_samples':3}
    agent.config=train['params']['config']
    agent.model=DualAMPModel.Network(net)
    return agent


def batch_inputs(agent,far=True,replay_far=True):
    count=8
    obs=torch.randn(count,1290);replay=torch.randn_like(obs);demo=torch.randn_like(obs)
    batch=dict(amp_cond=torch.full((4,2,10),2. if far else .2),
               amp_cond_replay=torch.full((4,2,10),2. if replay_far else .2),
               amp_obs_demo_masked=torch.randn(4,2,1290))
    return agent._prepare_amp_training_inputs(obs,replay,demo,batch),obs


def model_inputs(amp):
    poses=torch.zeros(2,6,7);poses[...,6]=1
    obs=torch.cat((torch.randn(2,508),poses.flatten(1)),-1)
    return dict(is_train=True,prev_actions=torch.zeros(4,32),obs=obs,**amp)


def test_masks_every_frame_without_mutating_full_input():
    x=torch.randn(3,10*129);before=x.clone();masked=mask_history(x).reshape(3,10,129)
    assert not masked[...,list(ARM_FEATURES)].any()
    torch.testing.assert_close(x,before)
    assert (x.reshape(3,10,129)[...,list(ARM_FEATURES)]!=0).all()


def test_blend_is_history_mean_sigmoid_and_full_outside_is_history_min():
    distances=torch.tensor([[0.,2.],[1.,1.],[2.,2.]])
    w=blend_weight(distances)
    torch.testing.assert_close(w[:2],torch.full((2,1),.5))
    assert w[-1]<1e-5
    assert outside_history(distances).tolist()==[False,False,True]
    torch.testing.assert_close(blend_rewards(torch.full((3,1),2.),torch.full((3,1),6.),distances),2+4*w)
    assert torch.isfinite(style_reward(torch.tensor([-100.,0.,100.]),2.)).all()


def test_independent_window_rms_sees_different_arm_statistics():
    agent=make_agent();x=torch.full((8,1290),3.)
    agent._preproc_amp_obs(x);agent._preproc_amp_obs_masked(x)
    full=agent._amp_input_mean_std.running_mean.reshape(10,129)
    masked=agent._amp_masked_input_mean_std.running_mean.reshape(10,129)
    assert (full[:,list(ARM_FEATURES)]>2.6).all()  # RunningMeanStd starts with a unit pseudo-count.
    assert not masked[:,list(ARM_FEATURES)].any()
    assert full.data_ptr()!=masked.data_ptr()


@pytest.mark.parametrize('far,replay_far',[(True,True),(False,True),(True,False)])
def test_both_losses_backward_or_full_safely_skipped(far,replay_far):
    agent=make_agent();amp,raw=batch_inputs(agent,far,replay_far)
    result=agent.model(model_inputs(amp))
    loss=agent._compute_discriminator_loss(result,amp)
    assert torch.isfinite(loss['disc_loss'])
    loss['disc_loss'].backward()
    net=agent.model.a2c_network
    assert net._disc_logits_masked.weight.grad is not None
    assert torch.isfinite(net._disc_logits_masked.weight.grad).all()
    assert (net._disc_logits.weight.grad is not None)==(far and replay_far)
    assert net._disc_logits_masked.weight.data_ptr()!=net._disc_logits.weight.data_ptr()
    assert loss['full_updated']==float(far and replay_far)
    # The full branch preserves arms; the masked normalizer zeroes all histories.
    assert not amp['amp_obs_masked'].reshape(-1,10,129)[...,list(ARM_FEATURES)].any()
    assert (raw.reshape(-1,10,129)[...,list(ARM_FEATURES)]!=0).any()


def test_full_branch_requires_enough_outside_samples():
    agent=make_agent();agent._dual_cfg['full_min_samples']=501
    amp,_=batch_inputs(agent,True,True)
    assert amp['amp_obs'] is None and amp['amp_obs_demo'] is None


def test_paired_replay_preserves_distances_through_capacity_and_keep_sampling():
    agent=object.__new__(DualAMPAgent);agent.ppo_device='cpu'
    agent._amp_replay_buffer=ReplayBuffer(8,'cpu');agent._amp_replay_keep_prob=.5
    for start in (0,20,40):
        ids=torch.arange(start,start+12).reshape(6,2,1).float()
        agent._store_amp_batch_replay({'amp_obs':ids.expand(-1,-1,3),'amp_cond':(ids+100).expand(-1,-1,10)})
        sample=agent._amp_replay_buffer.sample(8)
        torch.testing.assert_close(sample['amp_cond'][:,0]-sample['amp_obs'][:,0],torch.full((8,),100.))
    empty=object.__new__(DualAMPAgent);empty._amp_replay_buffer=ReplayBuffer(8,'cpu')
    data={'amp_obs':torch.arange(24).reshape(4,2,3),'amp_cond':torch.arange(80).reshape(4,2,10)}
    replay=empty._sample_amp_replay(data,8)
    torch.testing.assert_close(replay['amp_cond'],data['amp_cond'].reshape(8,10))


def test_scene_grouping_keeps_distance_history_matched():
    agent=object.__new__(DualAMPAgent);agent.num_actors=3;agent.num_agents=2
    ids=torch.arange(4*6).reshape(4,6,1).float()
    obs=ids.expand(-1,-1,1290);condition=(ids+100).expand(-1,-1,10)
    grouped=agent._group_agent_tensor(obs);dist=agent._group_agent_tensor(condition)
    torch.testing.assert_close(dist[...,0]-grouped[...,0],torch.full((12,2),100.))


def test_rim_distance_is_xy_nearest_edge_and_per_owner():
    box=torch.zeros(1,2,13);box[...,6]=1;box[0,1,1]=10
    roots=torch.tensor([[[3.,0.,100.],[3.,10.,0.]]]);size=torch.full((1,2,3),2.)
    torch.testing.assert_close(rim_distance(roots,box,size),torch.full((1,2),2.))
    roots[0,0,0]=1
    assert rim_distance(roots,box,size)[0,0]==0


def test_checkpoint_rejects_single_amp_missing_stats_and_config_mismatch():
    metadata={'version':1,'amp_shape':[1290]}
    with pytest.raises(ValueError):check_dual_checkpoint({},metadata)
    with pytest.raises(ValueError):check_dual_checkpoint({'dual_amp_metadata':metadata},metadata)
    state={'dual_amp_metadata':metadata,'amp_masked_input_mean_std':{}}
    check_dual_checkpoint(state,metadata)
    with pytest.raises(ValueError):check_dual_checkpoint(state,{'version':2})


def test_dual_model_checkpoint_restores_both_discriminators():
    a=make_agent();b=make_agent()
    with torch.no_grad():a.model.a2c_network._disc_logits_masked.weight.add_(1.)
    b.model.load_state_dict(a.model.state_dict())
    x=torch.randn(2,1290)
    torch.testing.assert_close(a.model.a2c_network.eval_disc(x),b.model.a2c_network.eval_disc(x))
    torch.testing.assert_close(a.model.a2c_network.eval_disc_masked(x),b.model.a2c_network.eval_disc_masked(x))


def test_single_amp_hook_preserves_normalization_order_and_inputs():
    from learning.multi_agent.ma_agent import MAAgent
    seen=[]
    def preprocess(value):
        seen.append(float(value[0,0]))
        return value+10
    dummy=SimpleNamespace(_preproc_amp_obs=preprocess)
    obs=torch.full((2,1290),1.);replay=torch.full_like(obs,2.);demo=torch.full_like(obs,3.)
    result=MAAgent._prepare_amp_training_inputs(dummy,obs,replay,demo,{})
    assert seen==[1.,2.,3.]
    assert result['amp_obs_demo'].requires_grad
    assert (result['amp_obs']==11).all() and (result['amp_obs_replay']==12).all()
    assert (obs==1).all() and (replay==2).all() and (demo==3).all()


def test_single_amp_replay_hook_matches_original_scene_order():
    from learning.multi_agent.ma_agent import MAAgent
    dummy=SimpleNamespace(_amp_replay_buffer=ReplayBuffer(8,'cpu'))
    obs=torch.arange(24).reshape(4,2,3).float()
    result=MAAgent._sample_amp_replay(dummy,{'amp_obs':obs},8)
    torch.testing.assert_close(result['amp_obs'],obs.reshape(8,3))
    dummy._amp_replay_buffer.store({'amp_obs':obs.reshape(8,3)})
    result=MAAgent._sample_amp_replay(dummy,{'amp_obs':obs},4)
    assert list(result)==['amp_obs'] and result['amp_obs'].shape==(4,3)
