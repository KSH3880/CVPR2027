"""Task behavior contracts: no reward farming, hand contact, AMP phases and binding."""
import copy
from pathlib import Path
import sys
import pytest
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tokenhsi'))
from utils.push_door_spec import (phase_update,progress_reward,advance_success,expert_time,
    task_relation_matrix,hand_handle_contact,validate_interaction,interaction_metadata,check_interaction_checkpoint)


def test_repeated_open_close_and_push_backtracking_cannot_farm_progress():
    best=torch.tensor([0.,-1.])
    earned=[]
    for value in ([.5,-.5],[0.,-1.],[.5,-.5],[.7,-.3]):
        reward,best=progress_reward(torch.tensor(value),best,1.)
        earned.append(reward)
    torch.testing.assert_close(earned[1],torch.zeros(2))
    torch.testing.assert_close(earned[2],torch.zeros(2))
    torch.testing.assert_close(sum(earned),torch.tensor([.7,.7]))


def test_contact_requires_the_nearby_hand_and_same_handle_force():
    hands=torch.tensor([[[0.,0.,0.],[3.,0.,0.]]])
    handles=torch.tensor([[[.05,0.,0.],[4.,0.,0.]]])
    hand_force=torch.tensor([[2.,0.]])
    handle_force=torch.tensor([[2.,0.]])
    assert hand_handle_contact(hands,handles,hand_force,handle_force,.16,.5).item()
    for hf,of in ((torch.zeros_like(hand_force),handle_force),(hand_force,torch.zeros_like(handle_force)),
                  (torch.tensor([[0.,2.]]),handle_force),(hand_force,torch.tensor([[0.,2.]]))):
        assert not hand_handle_contact(hands,handles,hf,of,.16,.5).item()
    assert not hand_handle_contact(hands+10,handles,hand_force,handle_force,.16,.5).item()


def test_success_requires_contiguous_hold_and_bonus_only_once():
    elapsed=torch.zeros(1);done=torch.zeros(1,dtype=torch.bool)
    for valid in (True,False,True):
        elapsed,done,first=advance_success(torch.tensor([valid]),elapsed,done,.5,1.)
        assert not first.item()
    elapsed,done,first=advance_success(torch.tensor([True]),elapsed,done,.5,1.)
    assert first.item() and done.item()
    elapsed,done,first=advance_success(torch.tensor([False]),elapsed,done,.5,1.)
    assert elapsed.item()==0 and done.item() and not first.item()
    for _ in range(2):elapsed,done,first=advance_success(torch.tensor([True]),elapsed,done,.5,1.)
    assert not first.item()


def test_phase_hysteresis_ignores_small_angle_fluctuations():
    phase=torch.tensor([False])
    observed=[]
    for angle in (79.,81.,78.,66.,64.,70.,82.):
        phase=phase_update(phase,torch.tensor([angle]),80.,65.);observed.append(phase.item())
    assert observed==[False,True,True,True,False,False,True]


def test_expert_history_stays_inside_each_phase():
    lengths=torch.tensor([2.,10.]);history=.3
    for phase in ([0.,.7],[.7,1.]):
        for uniform in (torch.zeros(2),torch.ones(2)):
            times=expert_time(lengths,history,phase,uniform)
            assert torch.all(times-history>=lengths*phase[0]-1e-6)
            assert torch.all(times<=lengths*phase[1]+1e-6)
    with pytest.raises(ValueError):expert_time(torch.tensor([.5]),history,[.7,1.])


def test_all_four_task_pairs_bind_correct_object_and_goal():
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    matrices=task_relation_matrix(tasks)
    for row in range(4):
        for owner in range(2):
            active=2+owner+2*tasks[row,owner]
            inactive=2+owner+2*(1-tasks[row,owner])
            assert matrices[row,owner,active]==3 and matrices[row,active,6+owner]==5
            assert matrices[row,owner,inactive]==0 and matrices[row,inactive,6+owner]==0
            assert matrices[row,owner,6+owner]==4
        assert matrices[row,0,1]==2
        assert torch.all(matrices[row].diag()==1)


def test_config_and_checkpoint_reject_changed_physics_reward_or_amp():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1.yaml').read_text())['env']
    validate_interaction(env['interaction'])
    metadata=interaction_metadata(env)
    check_interaction_checkpoint({'interaction_metadata':metadata},metadata)
    for section,key,value in [('door','hold_seconds',3.),('amp','hold_source','loco'),('push','goal_tolerance',.2)]:
        changed=copy.deepcopy(env);changed['interaction'][section][key]=value
        with pytest.raises(ValueError):check_interaction_checkpoint({'interaction_metadata':metadata},interaction_metadata(changed))
    with pytest.raises(ValueError):check_interaction_checkpoint({},metadata)
    bad=copy.deepcopy(env['interaction']);bad['door']['reopen_degrees']=90
    with pytest.raises(ValueError):validate_interaction(bad)


def test_unified_scene_encoder_task_binding_gradient_and_token_permutation():
    import torch.nn as nn
    from learning.multi_agent.amp_network_builder_ma import RelationEncoder
    from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
    sizes=[223,34,9]
    encoder=RelationEncoder(sizes,2,4,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias=True,
        relation_bias_mode='edge_mlp',relation_graph_spec={'template':'push_door_stage1_v1'},
        gta_cfg={'enable':True,'translation_scale':1.,'representation':'se3_direct_sum'})
    obs=torch.randn(4,656)
    goals=obs[:,582:600].view(4,2,9)
    goals[...,:2]=torch.nn.functional.one_hot(torch.tensor([[0,0],[0,1],[1,0],[1,1]]),2).float()
    obs[:,600:].view(4,8,7)[...,3:7]=torch.tensor([0.,0.,0.,1.])
    normalizer=SceneRunningMeanStd(sizes,[2,4,2],[223,30,0],7)
    normalizer.eval()
    normalized=normalizer(obs)
    torch.testing.assert_close(normalized[:,582:],obs[:,582:])
    base=encoder(normalized)
    shuffled=encoder(normalized,token_order=torch.tensor([4,7,1,3,0,6,5,2]))
    torch.testing.assert_close(base,shuffled,atol=2e-6,rtol=2e-5)
    base.square().mean().backward()
    assert encoder.edge_encoder.bias_projection.grad.abs().sum()>0


def test_random_start_clearance_and_target_distribution():
    from utils.push_door_spec import sample_start_layout
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_random_start.yaml').read_text())['env']
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]]).repeat(1024,1)
    box,goal,human,yaw,box_yaw,shift=sample_start_layout(tasks,env['startRandomization'],torch.tensor([.5,.45,.7]),torch.tensor([-1.6,1.6]))
    push=tasks==0;door=tasks==1
    assert torch.all((box[...,0]-human[...,0])[push]>=.65-1e-6)
    assert torch.all((shift[...,0]-human[...,0])[door]>=.55-1e-6)
    distance=torch.linalg.vector_norm(goal[...,:2]-box[...,:2],dim=-1)
    assert distance.min()>=.7-1e-6 and distance.max()<=1.1+1e-6
    assert torch.allclose(box[...,2],torch.full_like(box[...,2],.355))
    assert human[...,0].std()>.1 and yaw.std()>.1 and box_yaw.std()>.1
    assert (human[:,1,1]-human[:,0,1]).min()>1.9
    other=sample_start_layout(tasks,env['startRandomization'],torch.tensor([.5,.45,.7]),torch.tensor([-1.6,1.6]))
    assert not torch.equal(box,other[0])
    original=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1.yaml').read_text())['env']
    with pytest.raises(ValueError):
        check_interaction_checkpoint({'interaction_metadata':interaction_metadata(original)},interaction_metadata(env))


def test_push_cube_mass_and_spawn_clearance():
    from utils.push_door_spec import sample_start_layout, push_box_start_x
    for name in ('push_door_stage1', 'push_door_stage1_random_start'):
        env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
        box=env['interaction']['push']['box']
        assert box['size']==[1.1,1.1,1.1]
        assert abs(box['density']*1.1**3-15.75)<1e-6
        radius=(2*.55**2)**.5
        start=push_box_start_x(box['size'],env['interaction']['push']['target_distance'][1])
        assert start+env['interaction']['push']['target_distance'][1]+radius<=-.2+1e-6
        if 'startRandomization' in env:
            tasks=torch.zeros(4096,2,dtype=torch.long)
            b,g,h,_,_,shift=sample_start_layout(tasks,env['startRandomization'],torch.tensor(box['size']),torch.tensor([-1.6,1.6]))
            assert (b[...,2]-.555).abs().max()<1e-6
            assert (b[...,0]-h[...,0]).min()>=1.1-1e-6
            assert (g[...,0]-shift[...,0]+radius).max()<=-.2+1e-6


def test_left_open_door_motion_filter():
    from utils.push_door_spec import door_motion_matches
    for side in ('left','right'):
        for mirror in ('','_M'):
            path='dataset_bones_dooropen/motions/inside_door_handle_'+side+'_side_open_walk_R_001__A512'+mirror+'/phys_humanoid_v3/ref_motion.npy'
            assert door_motion_matches(path,'left_open') == ((side=='left') != bool(mirror))
            assert door_motion_matches(path,'all')
    assert not door_motion_matches('unknown/clip/phys_humanoid_v3/ref_motion.npy','left_open')
    with pytest.raises(ValueError):door_motion_matches('x','invalid')


def test_task_rsi_config_rejects_impossible_angles_and_probabilities():
    config=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml').read_text())['env']['interaction']
    validate_interaction(config)
    for key,value in [('door_early_fraction',1.1),('frame_clearance',-.01),('door_early_max_degrees',80.)]:
        broken=copy.deepcopy(config);broken['task_rsi'][key]=value
        with pytest.raises(ValueError):validate_interaction(broken)


def test_frame_clearance_detects_crossing_limb_between_safe_joints():
    from types import SimpleNamespace
    from utils.task_rsi import frame_clearance
    from tokenhsi.utils.door_asset import DoorSpec
    task=SimpleNamespace(_door_spec=DoorSpec())
    body=torch.full((1,15,3),-3.)
    body[...,2]=1.
    assert frame_clearance(task,body).item()>1.
    # Forearm endpoints lie clear on opposite sides of the right jamb;
    # checking joints alone misses the intervening capsule crossing the frame.
    body[0,3]=torch.tensor([-.4,.52,1.])
    body[0,4]=torch.tensor([.4,.52,1.])
    assert frame_clearance(task,body).item()<0.
