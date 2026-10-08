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
    for name in ('push_door_stage1', 'push_door_stage1_random_start', 'push_door_stage1_task_rsi'):
        env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
        box=env['interaction']['push']['box']
        size=1.3 if name.endswith('task_rsi') else 1.1
        assert box['size']==[size]*3
        assert abs(box['density']*size**3-30.)<1e-6
        radius=(2*(size/2)**2)**.5
        start=push_box_start_x(box['size'],env['interaction']['push']['target_distance'][1])
        assert start+env['interaction']['push']['target_distance'][1]+radius<=-.2+1e-6
        if 'startRandomization' in env:
            tasks=torch.zeros(4096,2,dtype=torch.long)
            b,g,h,_,_,shift=sample_start_layout(tasks,env['startRandomization'],torch.tensor(box['size']),torch.tensor([-1.6,1.6]))
            assert (b[...,2]-(size/2+.005)).abs().max()<1e-6
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


def test_push_alignment_brings_unequal_hand_reach_to_same_face():
    from utils.task_rsi import push_hand_normal
    hands=torch.tensor([[[.5,-.2],[.8,.2]],[[.5,.2],[.8,-.2]],[[.6,0.],[.6,0.]]])
    normal=push_hand_normal(hands)
    distances=(hands*normal[:,None,:]).sum(-1)
    torch.testing.assert_close(distances[:,0],distances[:,1])
    assert (distances>0).all()
    torch.testing.assert_close(normal.norm(dim=-1),torch.ones(3))


def test_door_shaping_blocks_body_push_delayed_credit_and_approach_farming():
    from utils.push_door_spec import door_shaping
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml').read_text())['env']['interaction']['door']
    t=lambda x:torch.tensor([x],dtype=torch.float32)
    def step(angle,best,prev,distance,closest,contact):
        return door_shaping(t(angle),t(best),t(prev),t(distance),t(closest),torch.tensor([contact]),.1,cfg)
    opening,best,approach,closest,closing=step(.2,0.,0.,.2,.5,False)
    assert opening.item()==0 and approach.item()==0 and closing.item()==0
    assert best.item()==pytest.approx(.2)
    # Later contact at the same angle must not claim the earlier body push.
    assert step(.2,.2,.2,.2,.2,True)[0].item()==0
    assert step(.25,.2,.2,.2,.2,True)[0].item()>0
    closed=step(.15,.25,.25,.2,.2,False)
    assert closed[-1].item()>0
    assert step(.25,.25,.15,.2,.2,True)[0].item()==0
    near=step(.1,.1,.1,.3,.5,False)
    assert near[2].item()>0 and near[3].item()==pytest.approx(.3)
    assert step(.1,.1,.1,.3,.3,False)[2].item()==0
    assert step(.1,.1,.1,.5,.3,False)[2].item()==0
    # RSI reset values produce no opening/approach/closing shaping.
    reset=step(.3,.3,.3,.1,.1,False)
    assert reset[0].item()==0 and reset[2].item()==0 and reset[4].item()==0
    assert step(.299,.3,.3,.1,.1,False)[-1].item()==0


def test_push_away_direction_preserves_door_and_clears_trajectory():
    from utils.push_door_spec import sample_start_layout, redirect_push_away
    for name in ('push_door_stage1_random_start', 'push_door_stage1_task_rsi'):
        env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
        assert env['interaction']['push']['direction']=='away_from_door'
        tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]]).repeat(1024,1)
        b,g,h,y,by,shift=sample_start_layout(tasks,env['startRandomization'],torch.tensor([1.1]*3),torch.tensor([-1.6,1.6]))
        boxes=torch.zeros(*tasks.shape,13);boxes[...,:3]=b
        boxes[...,5]=torch.sin(by/2);boxes[...,6]=torch.cos(by/2)
        q=torch.zeros(*tasks.shape,4);q[...,2]=torch.sin(y/2);q[...,3]=torch.cos(y/2)
        nb,ng,nh,nq=redirect_push_away(boxes,g,h,q,tasks)
        push=tasks==0;door=~push
        assert torch.all(ng[...,0][push]<b[...,0][push])
        assert torch.all(nh[...,0][push]>b[...,0][push])
        # Entire box-to-goal segment moves away from the door plane.
        assert torch.all(ng[...,0][push]<shift[...,0][push])
        torch.testing.assert_close(torch.linalg.vector_norm(ng[...,:2]-b[...,:2],dim=-1),torch.linalg.vector_norm(g[...,:2]-b[...,:2],dim=-1))
        for original,changed in ((boxes,nb),(g,ng),(h,nh),(q,nq)):
            torch.testing.assert_close(original[door],changed[door])
        torch.testing.assert_close(nb[...,:3],boxes[...,:3])
        torch.testing.assert_close(torch.linalg.vector_norm(nq,dim=-1),torch.ones_like(y))
        # RSI follows the box orientation when it regenerates a PUSH goal.
        yaw=2*torch.atan2(nb[...,5],nb[...,6])
        assert torch.all(torch.cos(yaw[push])<0)



def test_restored_reward_and_lower_handle_configuration():
    for name in ('push_door_stage1','push_door_stage1_random_start','push_door_stage1_task_rsi'):
        env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
        c=env['interaction'];validate_interaction(c)
        if name=='push_door_stage1_task_rsi':assert c['push']['hand_reward_weight']==.15
        else:assert 'hand_reward_weight' not in c['push']
        assert 'opening_weight' not in c['door'] and 'angle_weight' not in c['door']
        assert c['progress_weight']==.4 and c['door']['handle_height']==.95
        assert c['push']['direction']=='away_from_door'
        old=copy.deepcopy(env);old['interaction']['door'].pop('handle_height')
        with pytest.raises(ValueError):
            check_interaction_checkpoint({'interaction_metadata':interaction_metadata(old)},interaction_metadata(env))


def test_motion_contact_hand_tracks_mirror_and_rejects_ambiguous_clip():
    from utils.task_rsi import motion_contact_hand
    extension=torch.tensor([[.6,.2],[.5,.2],[.2,.6],[.2,.5],[.3,.31],[.32,.3]])
    ids=torch.tensor([0,0,1,1,2,2])
    assert motion_contact_hand(extension,ids,3,.06).tolist()==[0,1,-1]


def test_door_rsi_rejects_backwards_pelvis_or_torso_and_forces_motion_hand():
    from types import SimpleNamespace
    from tokenhsi.utils.door_asset import DoorSpec
    from utils.task_rsi import safe_door_candidates
    cfg={'min_extension':.2,'door_height_tolerance':.1,'door_angle_degrees':[5.,79.],
         'hand_gap':.03,'frame_clearance':.03,'door_facing_degrees':60.}
    task=SimpleNamespace(_interaction={'task_rsi':cfg},_key_body_ids=torch.tensor([5,8]),_door_spec=DoorSpec(handle_height=.95))
    root=torch.tensor([[0.,0.,1.]])
    body=root[:,None,:].expand(-1,15,-1).clone()
    body[:,5]=torch.tensor([.55,-.05,.95]);body[:,8]=torch.tensor([.3,.05,.95])
    identity=torch.tensor([[0.,0.,0.,1.]])
    back=torch.tensor([[0.,0.,1.,0.]])
    hand=torch.tensor([0])
    angles,q,origins,safe=safe_door_candidates(task,root,body,identity,identity,hand)
    assert safe.any()
    for pelvis,torso in ((back,identity),(identity,back),(back,back)):
        assert not safe_door_candidates(task,root,body,pelvis,torso,hand)[3].any()
    assert not safe_door_candidates(task,root,body,identity,identity,torch.tensor([-1]))[3].any()
    body[:,5,2]=1.5
    assert not safe_door_candidates(task,root,body,identity,identity,hand)[3].any()


def test_door_approach_contact_mix_and_far_spawn():
    from utils.task_rsi import select_task_rsi
    from utils.push_door_spec import sample_start_layout
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml').read_text())['env']
    config=env['interaction']['task_rsi'];validate_interaction(env['interaction'])
    torch.manual_seed(42);tasks=torch.tensor([[0,1]]).repeat(20000,1)
    selected=select_task_rsi(tasks,config)
    assert abs(selected[:,0].float().mean().item()-.8)<.02
    assert abs(selected[:,1].float().mean().item()-.5)<.02
    b,g,h,yaw,by,shift=sample_start_layout(tasks,env['startRandomization'],torch.tensor([1.1]*3),torch.tensor([-1.6,1.6]))
    distance=shift[:,1,0]-h[:,1,0]
    assert distance.min()>=1.2-1e-6 and distance.max()<=1.8+1e-6
    assert (h[:,1,1]-shift[:,1,1]+.36).abs().max()<=.2+1e-6
    # Legacy viewer configs keep their original single probability.
    legacy=select_task_rsi(tasks,{'probability':1.})
    assert legacy.all()
    assert not select_task_rsi(tasks,{'probability':0.,'door_rsi_probability':0.}).any()


def test_user_confirmed_door_motion_hands():
    from utils.push_door_spec import door_motion_hand
    for actor in ('A512','A513','A514','A515'):
        assert door_motion_hand('motions/inside_door_handle_left_side_open_walk_R_001__'+actor+'/phys_humanoid_v3/ref_motion.npy')==0
        assert door_motion_hand('motions/inside_door_handle_right_side_open_walk_R_001__'+actor+'_M/phys_humanoid_v3/ref_motion.npy')==1
    with pytest.raises(ValueError):
        door_motion_hand('motions/inside_door_handle_left_side_open_walk_R_001__A512_M/phys_humanoid_v3/ref_motion.npy')


def test_five_amp_families_match_hands_and_preserve_raw_labels():
    from utils.unified_training import sample_family_matched,preserve_amp_labels,amp_family_ids
    ids=torch.arange(5).repeat(3)
    labels=torch.nn.functional.one_hot(ids,5).float()
    frame=torch.cat([ids[:,None].float(),labels],-1)
    pool=frame[:,None,:].repeat(1,10,1).flatten(1)
    reference=pool.flip(0)
    matched=sample_family_matched(pool,reference,10,num_families=5)
    torch.testing.assert_close(amp_family_ids(matched,10,5),amp_family_ids(reference,10,5))
    torch.testing.assert_close(matched.reshape(-1,10,6)[:,0,0].long(),amp_family_ids(reference,10,5))
    normalized=preserve_amp_labels(pool,pool+42,10,5).reshape(-1,10,6)
    torch.testing.assert_close(normalized[:,:,1:],pool.reshape(-1,10,6)[:,:,1:])
    torch.testing.assert_close(normalized[:,:,0],pool.reshape(-1,10,6)[:,:,0]+42)
    with pytest.raises(ValueError):
        sample_family_matched(pool[ids!=4],reference,10,num_families=5)
    fallback=sample_family_matched(pool[ids!=4],reference,10,reference,5)
    torch.testing.assert_close(amp_family_ids(fallback,10,5),amp_family_ids(reference,10,5))


def test_door_amp_approach_distance_hysteresis_and_contact_priority():
    from utils.push_door_spec import door_open_amp_phase,door_amp_family
    t=lambda x:torch.tensor([x])
    phase=t(False)
    for distance,expected in ((1.5,False),(.81,False),(.8,True),(.9,True),(1.,True),(1.01,False)):
        phase=door_open_amp_phase(phase,t(distance),t(False),t(False),.8,1.)
        assert phase.item()==expected
    assert door_open_amp_phase(t(False),t(1.5),t(True),t(False),.8,1.).item()
    assert door_open_amp_phase(t(False),t(1.5),t(False),t(True),.8,1.).item()
    tasks=torch.tensor([0,1,1,1,1,1]);hands=torch.tensor([0,0,1,0,1,0])
    hold=torch.tensor([False,False,False,True,True,False])
    near=torch.tensor([False,True,True,False,False,False])
    assert door_amp_family(tasks,hands,hold,near,6).tolist()==[0,1,2,3,4,5]
    assert door_amp_family(tasks,hands,hold,near,5).tolist()==[0,1,2,3,4,1]


def test_six_amp_family_loco_demos_and_replay_do_not_mix_with_door():
    from utils.unified_training import sample_family_matched,preserve_amp_labels,amp_family_ids
    ids=torch.arange(6).repeat(3)
    frame=torch.cat([ids[:,None].float(),torch.nn.functional.one_hot(ids,6).float()],-1)
    pool=frame[:,None,:].repeat(1,10,1).flatten(1)
    matched=sample_family_matched(pool,pool.flip(0),10,num_families=6)
    torch.testing.assert_close(matched.reshape(-1,10,7)[:,0,0].long(),amp_family_ids(pool.flip(0),10,6))
    normalized=preserve_amp_labels(pool,pool+42,10,6).reshape(-1,10,7)
    torch.testing.assert_close(normalized[:,:,1:],pool.reshape(-1,10,7)[:,:,1:])


def test_restored_push_hands_bonus_rejects_lifted_and_single_hand():
    from utils.push_door_spec import push_hand_alignment
    cfg={'hand_radius':.04,'hand_distance_scale':.15,'hand_height_fraction':[.6,.95]}
    hands=torch.tensor([[[-.59,-.2,.2],[-.59,.2,.2]]]);direction=torch.tensor([[1.,0.,0.]])
    quality,error=push_hand_alignment(hands,direction,torch.tensor([.55]*3),cfg)
    torch.testing.assert_close(quality,torch.ones(1))
    for index in (0,1):
        raised=hands.clone();raised[:,index,2]=1.
        bad,_=push_hand_alignment(raised,direction,torch.tensor([.55]*3),cfg)
        assert bad.item()<.01
    backwards,_=push_hand_alignment(-hands,direction,torch.tensor([.55]*3),cfg)
    assert backwards.item()<.01


def test_push_box_size_override_preserves_mass_and_clearance():
    from utils.push_door_spec import resize_push_box, sample_start_layout
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml').read_text())['env']
    resize_push_box(env, 1.5)
    box=env['interaction']['push']['box']
    assert box['size']==[1.5]*3
    assert abs(box['density']*1.5**3-30)<1e-6
    assert env['startRandomization']['push_distance'][0]>=2**.5*.75+.25
    sample_start_layout(torch.zeros(16,2,dtype=torch.long),env['startRandomization'],torch.tensor(box['size']),torch.tensor([-1.6,1.6]))
    for invalid in (-1, float('nan'), float('inf'), 0):
        with pytest.raises(ValueError): resize_push_box(env, invalid)
