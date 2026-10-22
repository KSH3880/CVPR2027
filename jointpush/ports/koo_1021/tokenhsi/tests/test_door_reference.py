"""CPU tests of unified/shared comparison and screened Door proxy AMP/RSI."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
import yaml
from utils.door_reference import DoorReferenceBank, candidate_mask, align_reference
from utils.edge_door_spec import validate_env, compile_graph
from utils.edge_stage1_spec import semantic_graph_packet

ROOT=Path(__file__).resolve().parents[1]

def config(variant, suffix='amp_on'):
    name='door_open_'+variant+'_'+suffix+'.yaml'
    return (yaml.safe_load((ROOT/'data/cfg/multi_agent'/name).read_text())['env'],
        yaml.safe_load((ROOT/'data/cfg/train/rlg'/name).read_text()))

def test_unified_shared_differ_only_in_edge_parameter_sharing():
    a,at=config('unified');b,bt=config('shared_edge')
    assert a==b
    for env in (a,b):validate_env(env)
    at['params']['network']['transformer']['share_edge_encoder']=True
    assert at==bt
    old=yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())['params']['network']
    expected=deepcopy(old);expected['transformer']['share_edge_encoder']=False
    assert config('unified')[1]['params']['network']==expected

def fake_lib():
    n=20
    pos=torch.zeros(2*n,15,3);pos[:,0,2]=1.
    pos[:,[5,8],0]=.5;pos[:,[5,8],2]=1.
    pos[:,[11,14],2]=.1
    rot=torch.zeros(2*n,15,4);rot[...,3]=1.
    return SimpleNamespace(gts=pos,grs=rot,grvs=torch.zeros(2*n,3),
        length_starts=torch.tensor([0,n]),_motion_num_frames=torch.tensor([n,n]),
        _motion_dt=torch.tensor([.1,.1]),_motion_weights=torch.tensor([.25,.75]),num_motions=lambda:2)

def test_complete_history_and_clip_weights_late_bias():
    spec=deepcopy(config('unified')[0]['doorReference']);spec['interaction_skills']=['pickUp']
    lib=fake_lib();bank=DoorReferenceBank({'pickUp':lib},[5,8],[11,14],spec,.3)
    torch.manual_seed(14)
    ids,times=bank.sample('pickUp',5000,late=True)
    assert (times>=.3-1e-6).all()
    assert abs((ids==1).float().mean().item()-.75)<.03
    assert (times>=1.4-1e-6).float().mean()>.7
    lib.grvs[:]=10.
    with pytest.raises(ValueError,match='No geometry-valid'):
        DoorReferenceBank({'pickUp':lib},[5,8],[11,14],spec,.3)

@pytest.mark.parametrize('condition',['low_hand','fast_root','bent_root','high_feet','rear_hand'])
def test_excludes_unsuitable_interaction(condition):
    spec=config('unified')[0]['doorReference'];lib=fake_lib()
    if condition=='low_hand':lib.gts[:,[5,8],2]=.3
    if condition=='fast_root':lib.grvs[:,0]=2.
    if condition=='bent_root':lib.gts[:,0,2]=.5
    if condition=='high_feet':lib.gts[:,[11,14],2]=.5
    if condition=='rear_hand':lib.gts[:,[5,8],0]=-.3
    assert not candidate_mask(lib.gts,lib.grs[:,0],lib.grvs,[5,8],[11,14],spec).any()

def test_aligns_only_own_handle_and_keeps_reference_height():
    lib=fake_lib();pos=lib.gts[:2];rot=lib.grs[:2]
    target=torch.tensor([[1.,-2.6,1.],[1.,2.4,1.]])
    body,rotation,vel,ang,q=align_reference(pos,rot,torch.zeros_like(pos),torch.zeros_like(pos),rot[:,0],target,[5,8],.08)
    torch.testing.assert_close(body[:,5,:2],target[:,:2]+torch.tensor([0.,-.08]))
    torch.testing.assert_close(body[:,0,2],pos[:,0,2])
    changed=target.clone();changed[1,:2]+=100
    other=align_reference(pos,rot,torch.zeros_like(pos),torch.zeros_like(pos),rot[:,0],changed,[5,8],.08)[0]
    torch.testing.assert_close(body[0],other[0])

@pytest.mark.parametrize('suffix',['amp_on','approach_amp'])
@pytest.mark.parametrize('variant',['unified','shared_edge'])
def test_actual_network_sharing_and_actor_critic_gradients(variant,suffix):
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    torch.set_num_threads(1)
    env,train=config(variant,suffix)
    builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
    net=builder.build('amp',actions_num=28,input_shape=(570,),amp_input_shape=(1290,),
        value_size=1,num_agents=2,num_objects=2,humanoid_obs_size=230,object_obs_size=39,
        goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,35,1],
        scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode=env['relationReward']['mode'],
        relation_graph_spec=env['relationGraph'],device='cpu')
    actor,critic=net.actor_encoder,net.critic_encoder
    shared=variant=='shared_edge'
    assert (actor.edge_encoder is critic.edge_encoder)==shared
    assert actor.tokenizers is not critic.tokenizers and actor.layers is not critic.layers
    with torch.no_grad():
        actor.edge_encoder.bias_projection.normal_(std=.01)
        critic.edge_encoder.bias_projection.normal_(std=.01)
    poses=torch.zeros(2,6,7);poses[...,6]=1.
    obs=torch.cat((torch.randn(2,518),poses.flatten(1),semantic_graph_packet(compile_graph(env['relationGraph'],2,2),2)),-1)
    for encoder,head,other in [(actor,net.action_head,critic),(critic,net.value_head,actor)]:
        encoder.gta_diagnostics_first_forward=False
        net.zero_grad();output=head(encoder(obs));assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert encoder.edge_encoder.bias_projection.grad.abs().sum()>0
        assert (other.edge_encoder.bias_projection.grad is not None)==shared

def test_reference_yaml_has_only_available_proxy_skills():
    manifest=yaml.safe_load((ROOT/'data/dataset_door_proxy.yaml').read_text())
    assert set(manifest['motions'])=={'loco','pickUp','carryWith'}
    for rows in manifest['motions'].values():
        for row in rows:assert (ROOT/'data'/row['file']).is_file()


def test_late_sampling_with_no_late_request():
    spec=deepcopy(config('unified')[0]['doorReference'])
    spec['interaction_skills']=['pickUp'];spec['late_probability']=0.
    bank=DoorReferenceBank({'pickUp':fake_lib()},[5,8],[11,14],spec,.3)
    ids,times=bank.sample('pickUp',1,late=True)
    assert ids.shape==times.shape==(1,)
