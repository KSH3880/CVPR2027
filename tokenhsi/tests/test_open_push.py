"""Joint task graph, grounded/contact-aware rewards and conditional AMP tests."""
from copy import deepcopy
from pathlib import Path
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from utils.open_push_spec import validate_env, graph_for_tasks, PUSH
from utils.edge_stage1_spec import compile_stage1_graph, semantic_graph_packet
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata
from utils.unified_training import append_family, sample_family_matched, preserve_amp_labels
from env.tasks.multi_agent.open_push_reward import push_reward, push_hand_contact, OpenPushRuntime
ROOT=Path(__file__).resolve().parents[1]

def env():return yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_unified_amp.yaml').read_text())['env']

def geometry(n=1):
    boxes=torch.zeros(n,2,13);boxes[...,2]=env()['pushBox']['size'][2]/2;boxes[...,6]=1
    start=boxes[...,:3].clone();goals=start.clone();goals[...,1]=1.5
    roots=start.clone();roots[...,1]-=1
    hands=start[:,:,None].expand(n,2,2,3).clone();hands[...,1]-=env()['pushBox']['size'][1]/2
    hands[...,0]=torch.tensor([.2,-.2]);hands[...,2]=env()['relationReward']['push']['hand_target_height']
    return roots,hands,boxes,goals,start

def test_config_manifest_and_taxonomy():
    e=env();validate_env(e);validate_relation_config(e['relationReward'])
    assert e['numEnvs']==2048 and e['relationGraph']['task_probabilities']=={'OPEN':.5,'PUSH':.5}
    assert checkpoint_metadata(e['relationReward'])['relation_taxonomy']=={'open':11,'push':12,'at':7}
    g=compile_stage1_graph(e['relationGraph'],2,4)
    assert len(g.ids)==4
    path=ROOT/'data/dataset_open_push.yaml';m=yaml.safe_load(path.read_text())['motions']
    assert {k:len(v) for k,v in m.items()}=={'loco':4,'doorOpen':1,'push':2}
    for entries in m.values():
        for entry in entries:assert (path.parent/entry['file']).is_file()


def test_debug_distinguishes_rocking_from_bottom_slide_and_rsi_offset():
    from env.tasks.multi_agent.open_push_reward import push_debug_metrics
    e=env();roots,hands,boxes,goals,start=geometry()
    spec=e['relationReward']['push'];size=e['pushBox']['size']
    theta=torch.tensor(.2);depth=size[1]/2;height=size[2]/2
    boxes[...,3]=-torch.sin(theta/2);boxes[...,6]=torch.cos(theta/2)
    boxes[...,1]=height*torch.sin(theta)+depth*(1-torch.cos(theta))
    boxes[...,2]=height*torch.cos(theta)+depth*torch.sin(theta)
    forces=torch.zeros_like(hands);bf=torch.zeros(1,2,3);flags=torch.zeros(1,2,dtype=torch.bool)
    diag=push_debug_metrics(hands,forces,boxes,bf,start,size,spec,flags,flags)
    assert (diag['box_forward_m']>.1).all()
    torch.testing.assert_close(diag['bottom_edge_forward_m'],torch.zeros(1,2),atol=1e-6,rtol=0)
    torch.testing.assert_close(diag['tilt_rad'],torch.full((1,2),.2))
    episode=boxes[...,:3].clone();boxes[...,1]+=.05
    diag=push_debug_metrics(hands,forces,boxes,bf,start,size,spec,flags,flags,episode)
    torch.testing.assert_close(diag['since_reset_forward_m'],torch.full((1,2),.05))


def test_empty_and_populated_reset_diagnostics_are_tensorboard_scalars():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
    for data in ({},{'push/rsi/retarget_count':3,'push/rsi/right_reset_distance_sum':torch.tensor(.3)}):
        task=SimpleNamespace(_diag=data,_joint_rsi_spec={'enabled':True},device='cpu')
        result=HumanoidMAOpenPush.consume_relation_diagnostics(task)
        assert all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in result.values())
        assert result['push/rsi/retarget_count'].item() in (0,3)

@pytest.mark.parametrize('tasks',[[0,0],[0,1],[1,0],[1,1]])
def test_task_graph_and_goals(tasks):
    g=graph_for_tasks(env()['relationGraph'],torch.tensor([tasks]))
    assert semantic_graph_packet(g,1).shape==(1,20)
    for a,t in enumerate(tasks):
        assert g.edge_dst[0,2*a]==2+a+2*t
        assert g.edge_relation[0,2*a]==(PUSH if t else 11)
        assert g.edge_valid[0,2*a+1]==bool(t)
        assert g.edge_src[0,2*a+1]==4+a and g.edge_dst[0,2*a+1]==6+a

@pytest.mark.parametrize('field',['cross_expert','amp','counts','nan','rsi'])
def test_reject_invalid_contract(field):
    e=env()
    if field=='cross_expert':e['ampExpertMapping']['PUSH']=[.3,.7,0.]
    if field=='amp':e['ampTaskConditioning']=False
    if field=='counts':e['numObjects']=2
    if field=='nan':e['relationGraph']['task_probabilities']={'OPEN':float('nan'),'PUSH':.5}
    if field=='rsi':e['skillInitProb']=[0.,0.,1.]
    with pytest.raises(ValueError):validate_env(e)

def test_goal_requires_contact_ground_stability_and_current_success():
    e=env();roots,hands,boxes,goals,start=geometry()
    eligible=torch.ones(1,2,dtype=torch.bool)
    def reward():return push_reward(roots,hands,boxes,goals,start,eligible,e['relationReward'],e['pushBox']['size'])
    initial=reward();boxes[...,1]=.75;middle=reward();assert (middle['progress']>initial['progress']).all()
    boxes[...,:3]=goals;assert reward()['success'].all()
    eligible[:]=False;assert not reward()['success'].any();eligible[:]=True
    boxes[...,2]+=.2;assert not reward()['success'].any() and (reward()['progress']==0).all();boxes[...,2]-=.2
    boxes[...,3]=.5;boxes[...,6]=3**.5/2;assert not reward()['success'].any();boxes[...,3]=0;boxes[...,6]=1
    boxes[...,7]=1;assert not reward()['success'].any();boxes[...,7]=0
    boxes[...,10]=1;assert not reward()['success'].any();boxes[...,10]=0
    boxes[...,1]=1.;assert not reward()['success'].any()

def test_own_hand_contact_requires_force_and_rear_surface():
    e=env();roots,hands,boxes,goals,start=geometry()
    force=torch.zeros_like(hands);force[...,1]=-2
    bf=torch.zeros(1,2,3);bf[...,1]=2
    def contact():return push_hand_contact(hands,force,boxes,bf,e['pushBox']['size'],e['relationReward']['push'])
    assert contact().all()
    force[...,1]=2;assert not contact().any();force[...,1]=-2
    hands[:,0,:,0]+=2;assert not contact()[0,0] and contact()[0,1]


def test_push_requires_both_assigned_points_and_both_hand_forces():
    from env.tasks.multi_agent.open_push_reward import push_hand_targets
    e=env();roots,hands,boxes,goals,start=geometry()
    force=torch.zeros_like(hands);force[...,1]=-2
    bf=torch.zeros(1,2,3);bf[...,1]=2
    spec=e['relationReward']['push'];size=e['pushBox']['size']
    targets=push_hand_targets(boxes,size,spec)
    torch.testing.assert_close(targets,hands)
    eligible=torch.zeros(1,2,dtype=torch.bool)
    def reward():return push_reward(roots,hands,boxes,goals,start,eligible,e['relationReward'],size)['state']
    before=reward();hands[:,:,1,0]+=.5
    assert (reward()<before).all()
    assert not push_hand_contact(hands,force,boxes,bf,size,spec).any()
    hands.copy_(targets);force[:,:,1]=0
    assert not push_hand_contact(hands,force,boxes,bf,size,spec).any()
    force[...,1]=-2
    assert push_hand_contact(hands,force,boxes,bf,size,spec).all()
    hands.copy_(targets.flip(-2))
    assert not push_hand_contact(hands,force,boxes,bf,size,spec).any()
    # Rotating the box rotates both targets and preserves their fixed assignment.
    boxes[...,5]=2**-.5;boxes[...,6]=2**-.5
    rotated=push_hand_targets(boxes,size,spec)
    torch.testing.assert_close(rotated[...,2],torch.full((1,2,2),spec['hand_target_height']))
    torch.testing.assert_close(rotated[:,:,0,:2],boxes[...,:2]+torch.tensor([size[1]/2,.2]))

def test_runtime_partial_reset_and_airborne_contact_cannot_unlock_push():
    e=env();rt=OpenPushRuntime(3,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'])
    roots,hands,boxes,goals,start=geometry(3);ids=torch.arange(3)
    rt.reset(ids,torch.tensor([[0,1],[1,0],[1,1]]),boxes)
    handles=torch.zeros(3,2,3);angles=torch.zeros(3,2)
    rt.push_contact[:]=True;boxes[...,1]=.1;boxes[...,2]+=1
    rt.step(roots,hands,handles,angles,boxes,goals)
    assert not rt.push_eligible.any()
    boxes[...,2]-=1;rt.step(roots,hands,handles,angles,boxes,goals)
    assert torch.equal(rt.push_eligible,rt.tasks==1)
    boxes[...,:3]=goals;rt.step(roots,hands,handles,angles,boxes,goals)
    assert torch.equal(rt.done,rt.tasks==1)
    before=rt.suffix().clone();rt.reset(ids[1:2],torch.tensor([[0,0]]),boxes[1:2])
    torch.testing.assert_close(rt.suffix()[[0,2]],before[[0,2]])
    assert not rt.done[1].any() and rt.done[2].all()
    boxes[...,1]-=.4;rt.step(roots,hands,handles,angles,boxes,goals);assert not rt.done.any()

def test_two_family_expert_replay_matching_and_literal_labels():
    raw=append_family(torch.tensor([[10.],[20.],[30.],[40.]]),torch.tensor([0,1,0,1]),2).repeat(1,2)
    matched=sample_family_matched(raw,raw.flip(0),2,num_families=2)
    torch.testing.assert_close(matched.reshape(4,2,3)[...,-2:],raw.flip(0).reshape(4,2,3)[...,-2:])
    normalized=preserve_amp_labels(raw,raw*3,2,2)
    torch.testing.assert_close(normalized.reshape(4,2,3)[...,-2:],raw.reshape(4,2,3)[...,-2:])
    with pytest.raises(ValueError):sample_family_matched(raw[[0,2]],raw,2,num_families=2)

@pytest.mark.parametrize('name,frames',[
    ('push_obstacle_180_101__A341',367),('push_obstacle_180_003__A342',352)])
def test_push_conversion_has_valid_dofs_ground_and_forward_motion(name,frames):
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation
    from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
    from tokenhsi.scripts.multi_agent.convert_bones_door import DOF_BODIES,DOF_OFFSETS,joint_bounds
    path=ROOT/'data/dataset_bones_push/motions'/name/'phys_humanoid_v3'
    motion=SkeletonMotion.from_file(str(path/'ref_motion.npy'))
    audit=json.loads((path/'conversion_report.json').read_text())
    q=motion.local_rotation.numpy();assert q.shape==(frames,15,4) and motion.fps==30
    np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1,atol=1e-5)
    for values in (motion.global_translation,motion.global_velocity,motion.global_angular_velocity):assert np.isfinite(values.numpy()).all()
    rv=Rotation.from_quat(q.reshape(-1,4)).as_rotvec().reshape(-1,15,3)
    dof=np.concatenate([rv[:,b] if hi-lo==3 else rv[:,b,1:2] for b,lo,hi in zip(DOF_BODIES,DOF_OFFSETS[:-1],DOF_OFFSETS[1:])],axis=-1)
    lo,hi=joint_bounds(motion.skeleton_tree)
    assert (dof>=lo-1e-5).all() and (dof<=hi+1e-5).all()
    assert np.abs(np.diff(dof,axis=0)*30).max()<10
    assert audit['foot_mesh_floor_min_m']>=0 and audit['root_net_horizontal_m']>3
    assert max(audit['key_body_fit_error_max_m'])<.02


@pytest.mark.parametrize('tasks',[[0,0],[0,1],[1,0],[1,1]])
def test_joint_token_shuffle_preserves_actor_critic_outputs_and_gradients(tasks):
    from unittest.mock import patch
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    torch.manual_seed(117);torch.set_num_threads(1)
    e=env();assert e['relationGraph']['shuffle_token_order'] is True
    params=yaml.safe_load((ROOT/'data/cfg/train/rlg/open_push_unified_amp.yaml').read_text())['params']['network']
    builder=AMPMultiAgentBuilder();builder.load(params)
    network=builder.build('amp',actions_num=32,input_shape=(684,),amp_input_shape=(1310,),value_size=1,
        num_agents=2,num_objects=4,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
        observation_mode='clean_scene',scene_entity_sizes=[223,40,1],scene_kinematic_size=7,
        scene_arena_scale=7.,relation_reward_mode=e['relationReward']['mode'],
        relation_graph_spec=e['relationGraph'],device='cpu')
    network.eval()
    graph=graph_for_tasks(e['relationGraph'],torch.tensor([tasks,tasks]))
    kin=torch.randn(2,8,7);kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
    obs=torch.cat((torch.randn(2,608),kin.flatten(1),semantic_graph_packet(graph,2)),-1)
    permutation=torch.tensor([6,4,1,3,7,0,5,2])
    for encoder,head in ((network.actor_encoder,network.action_head),(network.critic_encoder,network.value_head)):
        encoder.gta_diagnostics_first_forward=False
        y=head(encoder(obs,token_order=torch.arange(8)));y.square().mean().backward()
        assert 'edge_bias/open/rms' in encoder.last_diagnostics
        assert encoder.last_diagnostics['edge_bias/push/sample_count'] == 2 * tasks.count(1)
        grads={n:p.grad.clone() for n,p in encoder.named_parameters() if p.grad is not None}
        encoder.zero_grad();head.zero_grad()
        z=head(encoder(obs,token_order=permutation));torch.testing.assert_close(y,z,atol=2e-5,rtol=2e-5)
        z.square().mean().backward()
        for n,p in encoder.named_parameters():
            if n in grads:torch.testing.assert_close(grads[n],p.grad,atol=5e-5,rtol=1e-3)
        with patch('torch.randperm',return_value=permutation) as shuffle:
            auto=head(encoder(obs));torch.testing.assert_close(y,auto,atol=2e-5,rtol=2e-5)
            shuffle.assert_called_once_with(8,device=obs.device)
        encoder.zero_grad();head.zero_grad()


def test_xy_approach_is_separate_from_box_goal_progress():
    from env.tasks.multi_agent.open_push_reward import approach_xy_diagnostics
    e=env();roots,hands,boxes,goals,start=geometry()
    handles=start.clone()
    before=approach_xy_diagnostics(roots,handles,boxes,e['relationReward'])
    roots[...,1]+=.5
    after=approach_xy_diagnostics(roots,handles,boxes,e['relationReward'])
    for name in ('open','push'):
        assert (after[name]['xy_progress'] > before[name]['xy_progress']).all()
        torch.testing.assert_close(after[name]['distance_xy'],torch.full((1,2),.5))
    assert not push_reward(roots,hands,boxes,goals,start,torch.zeros(1,2,dtype=torch.bool),
        e['relationReward'],e['pushBox']['size'])['progress'].any()


@pytest.mark.parametrize('distance,score',[(0.,1.),(.2,1.),(.5,1.),(1.,2/3),(1.5,.5)])
def test_push_approach_pins_at_half_meter_in_reward_and_diagnostics(distance,score):
    from env.tasks.multi_agent.open_push_reward import approach_xy_diagnostics
    e=env();roots,hands,boxes,goals,start=geometry()
    roots[...,:2]=boxes[...,:2];roots[...,1]-=distance
    expected=torch.full((1,2),score)
    diag=approach_xy_diagnostics(roots,start,boxes,e['relationReward'])
    for name in ('open','push'):
        torch.testing.assert_close(diag[name]['xy_progress'],expected)
    result=push_reward(roots,hands,boxes,goals,start,torch.zeros(1,2,dtype=torch.bool),
        e['relationReward'],e['pushBox']['size'])
    # Both hands are exactly on their targets, giving interaction score 1.
    torch.testing.assert_close(result['state'],.2*expected+.2)
    assert not result['progress'].any() and not result['success'].any()


def test_bias_diagnostics_use_active_edges_and_signed_logits():
    from learning.multi_agent.edge_context_encoder import PackedEdgeSemanticFusion
    g=graph_for_tasks(env()['relationGraph'],torch.tensor([[0,1],[1,0]]))
    bias=torch.full((2,2,2,8,8),99.)
    for n in range(2):
        for edge in range(4):
            if g.edge_valid[n,edge]:
                relation=int(g.edge_relation[n,edge])
                bias[:,n,:,g.edge_src[n,edge],g.edge_dst[n,edge]]={11:-2.,12:3.,7:4.}[relation]
    bias.requires_grad_()
    d=PackedEdgeSemanticFusion().bias_diagnostics(semantic_graph_packet(g,2),bias)
    for name,value in [('open',-2.),('push',3.),('at',4.)]:
        assert d['edge_bias/'+name+'/mean']==value
        assert d['edge_bias/'+name+'/rms']==abs(value)
        assert d['edge_bias/'+name+'/sample_count']==2
        assert not d['edge_bias/'+name+'/rms'].requires_grad
    assert d['edge_bias/holding/sample_count']==0
    assert d['edge_bias/holding/rms']==0
