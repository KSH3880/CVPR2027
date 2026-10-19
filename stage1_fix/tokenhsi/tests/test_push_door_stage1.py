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
    for name in ('push_door_stage1', 'push_door_stage1_random_start', 'push_door_stage1_task_rsi', 'push_door_stage1_task_rsi_variable_box'):
        env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
        box=env['interaction']['push']['box']
        size=1.3
        assert 'size_range' not in box
        assert box['size']==[size]*3
        assert abs(box['density']*size**3-25.)<1e-6
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
        assert 'hand_reward_weight' not in c['push']
        assert c['push']['relation_rewards']['hand_progress_weight']==.15
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
    cfg['door_alignment_yaw_degrees']=[-45.,0.,45.]
    multi=safe_door_candidates(task,root,body,identity,identity,hand)
    assert multi[0].numel()==64*3 and multi[3].any()
    chunked=safe_door_candidates(task,root.expand(65,-1),body.expand(65,-1,-1),identity.expand(65,-1),identity.expand(65,-1),hand.expand(65))
    torch.testing.assert_close(chunked[3],multi[3].expand(65,-1))
    cfg.pop('door_alignment_yaw_degrees')
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
    assert not selected[:,0].any() # low-hand PUSH RSI is disabled after physics validation
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
    assert abs(box['density']*1.5**3-25)<1e-6
    assert env['startRandomization']['push_distance'][0]>=2**.5*.75+.25
    sample_start_layout(torch.zeros(16,2,dtype=torch.long),env['startRandomization'],torch.tensor(box['size']),torch.tensor([-1.6,1.6]))
    for invalid in (-1, float('nan'), float('inf'), 0):
        with pytest.raises(ValueError): resize_push_box(env, invalid)


def test_nearest_door_hand_entry_preserves_binding_and_contact():
    from utils.push_door_spec import nearest_door_hand_on_entry
    hands=torch.tensor([[[2.,0.,0.],[.1,0.,0.]]]).expand(5,-1,-1)
    handles=torch.zeros(5,2,3)
    result=nearest_door_hand_on_entry(torch.zeros(5,dtype=torch.long),
        torch.tensor([False,True,False,False,False]),
        torch.tensor([True,True,False,True,True]),torch.tensor([1,1,1,0,1]),
        hands,handles,torch.tensor([False,False,False,False,True]))
    assert result.tolist()==[1,0,0,0,0]
    # No switching while staying near, even if the other hand becomes closer.
    assert nearest_door_hand_on_entry(result,torch.ones(5,dtype=torch.bool),
        torch.ones(5,dtype=torch.bool),torch.ones(5,dtype=torch.long),
        hands.flip(1),handles,torch.zeros(5,dtype=torch.bool)).tolist()==result.tolist()


def test_variable_box_wide_goal_distribution():
    from utils.push_door_spec import sample_start_layout
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi_variable_box.yaml').read_text())['env']
    validate_interaction(env['interaction'])
    with torch.random.fork_rng():
        torch.manual_seed(42)
        tasks=torch.zeros(4096,2,dtype=torch.long)
        sizes=(.9+.5*torch.rand(4096,2,1)).expand(-1,-1,3)
        box,goal,*_=sample_start_layout(tasks,env['startRandomization'],sizes,torch.tensor([-1.6,1.6]))
        delta=goal[...,:2]-box[...,:2]
        distance=delta.norm(dim=-1)
        angle=torch.rad2deg(torch.atan2(delta[...,1],delta[...,0]))
        assert distance.min()>=2-1e-5 and distance.max()<=4+1e-5
        assert angle.abs().max()<=60+1e-4
        assert (angle>40).float().mean()>.1
        assert (angle<-40).float().mean()>.1
        assert (delta[...,1].abs()>2).float().mean()>.1
        torch.testing.assert_close(box[...,2],sizes[...,2]/2+.005)
    assert env['interaction']['task_rsi']['push_goal_yaw_degrees']==[-60,60]


def test_full_circle_push_preserves_geometry_and_door():
    from utils.push_door_spec import rotate_push_layout
    tasks=torch.tensor([[0,1]]).repeat(1024,1)
    boxes=torch.zeros(1024,2,13);boxes[...,6]=1
    boxes[...,0]=-5.5;boxes[:,1,1]=12
    goals=boxes[...,:3].clone();goals[...,0]+=3
    humans=boxes[...,:3].clone();humans[...,0]-=1.5
    rotations=boxes[...,3:7].clone()
    angles=torch.linspace(-torch.pi,torch.pi,1024)[:,None].expand(-1,2)
    b,g,h,q=rotate_push_layout(boxes,goals,humans,rotations,tasks,angles)
    torch.testing.assert_close((g-b[...,:3]).norm(dim=-1),torch.full((1024,2),3.))
    torch.testing.assert_close((h-b[...,:3]).norm(dim=-1),torch.full((1024,2),1.5))
    for original,result in ((boxes,b),(goals,g),(humans,h),(rotations,q)):
        torch.testing.assert_close(original[:,1],result[:,1])
    delta=g[:,0,:2]-b[:,0,:2]
    assert all(((delta[:,0]*x>0)&(delta[:,1]*y>0)).any() for x,y in ((1,1),(1,-1),(-1,1),(-1,-1)))
    torch.testing.assert_close(q.norm(dim=-1),torch.ones(1024,2))


def test_semantic_push_door_edges_and_checkpoint_contract():
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    hold=torch.tensor([[False,False],[False,True],[False,False],[True,False]])
    matrix=task_relation_matrix(tasks,hold,semantic=True)
    for row in range(4):
        for owner in range(2):
            obj=2+owner+2*int(tasks[row,owner]);goal=6+owner
            expected=11 if tasks[row,owner]==0 else 13 if hold[row,owner] else 12
            assert matrix[row,owner,obj]==expected
            assert matrix[row,obj,goal]==(7 if tasks[row,owner]==0 else expected)
            assert matrix[row,obj,owner]==matrix[row,owner,obj]
            assert matrix[row,goal,obj]==matrix[row,obj,goal]
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi_variable_box.yaml').read_text())['env']
    legacy=copy.deepcopy(env);legacy['relationGraph'].pop('semantic_relations')
    with pytest.raises(ValueError):
        check_interaction_checkpoint({'interaction_metadata':interaction_metadata(legacy)},interaction_metadata(env))


def test_semantic_relations_reach_attention_with_gradients():
    import torch.nn as nn
    from learning.multi_agent.amp_network_builder_ma import RelationEncoder
    encoder=RelationEncoder([223,34,9],2,4,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias=True,
        relation_bias_mode='edge_mlp',relation_graph_spec={'template':'push_door_stage1_v1','semantic_relations':True},
        gta_cfg={'enable':True,'translation_scale':1.,'representation':'se3_direct_sum'})
    assert encoder.edge_encoder.relation_embed.num_embeddings==14
    obs=torch.randn(4,656)
    goals=obs[:,582:600].view(4,2,9)
    goals[...,:2]=torch.nn.functional.one_hot(torch.tensor([[0,0],[0,1],[1,0],[1,1]]),2).float()
    goals[...,2]=torch.tensor([[0.,0.],[0.,1.],[0.,0.],[1.,0.]])
    obs[:,600:].view(4,8,7)[...,3:7]=torch.tensor([0.,0.,0.,1.])
    seen=[]
    hook=encoder.edge_encoder.register_forward_pre_hook(lambda module,args:seen.append(args[1].clone()))
    with torch.no_grad():encoder.edge_encoder.bias_projection.normal_(std=.02)
    result=encoder(obs)
    result.square().sum().backward();hook.remove()
    assert {7,11,12,13}.issubset(set(seen[-1].unique().tolist()))
    grad=encoder.edge_encoder.relation_embed.weight.grad
    assert all(grad[index].abs().sum()>0 for index in (7,11,12,13))
    shuffled=encoder(obs,token_order=torch.tensor([4,7,1,3,0,6,5,2]))
    torch.testing.assert_close(result,shuffled,atol=2e-6,rtol=2e-5)


def test_split_push_relations_independent_rewards_and_no_repeat_results():
    from utils.push_door_spec import push_relation_rewards
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi_variable_box.yaml').read_text())['env']['interaction']['push']['relation_rewards']
    quality=torch.tensor([.8]);distance=torch.tensor([3.]);stable=torch.tensor([True]);contact=torch.tensor([False])
    best=torch.tensor([.2]);elapsed=torch.zeros(1);done=torch.tensor([False])
    terms,best,elapsed,done=push_relation_rewards(quality,contact,distance,stable,torch.zeros(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    assert terms[0,0]>0 and terms[0,1]>0 # hands can learn before the box moves
    assert terms[0,2]==0 and terms[0,4]==0 and terms[0,5]==0
    repeated,*_=push_relation_rewards(quality,contact,distance,stable,torch.zeros(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    assert repeated[0,1]==0 # no progress for waiting or returning to old quality
    contact[:]=True
    for _ in range(2):
        terms,best,elapsed,done=push_relation_rewards(quality,contact,distance,stable,torch.ones(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    assert terms[0,2]==cfg['hand_result_weight'] and terms[0,4]==cfg['at_progress_weight']
    contact[:]=False
    _,best,elapsed,done=push_relation_rewards(quality,contact,distance,stable,torch.ones(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    for _ in range(2):
        terms,best,elapsed,done=push_relation_rewards(quality,torch.tensor([True]),distance,stable,torch.ones(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    assert terms[0,2]==0 # releasing and touching again cannot farm result
    goal,*_=push_relation_rewards(quality,contact,torch.zeros(1),stable,torch.zeros(1),torch.tensor([True]),best,elapsed,done,.1,cfg)
    assert goal[0,3]==cfg['at_state_weight'] and goal[0,5]==cfg['at_result_weight']
    bad,*_=push_relation_rewards(quality,contact,distance,torch.tensor([False]),torch.ones(1),torch.tensor([False]),best,elapsed,done,.1,cfg)
    assert bad.eq(0).all()


def test_push_surface_distance_handles_size_and_rotation():
    from utils.push_door_spec import push_surface_distance
    boxes=torch.zeros(3,13);boxes[:,6]=1
    sizes=torch.tensor([[.9,.9,.9],[1.4,1.4,1.4],[2.,1.,1.]])
    roots=torch.tensor([[1.45,0.,1.],[1.7,0.,1.],[0.,2.,1.]])
    boxes[2,5:7]=torch.tensor([2**-.5,2**-.5]) # rotate long edge to Y
    torch.testing.assert_close(push_surface_distance(roots,boxes,sizes),torch.ones(3))


def test_push_loco_amp_switch_and_legacy_compatibility():
    from utils.push_door_spec import door_open_amp_phase,door_amp_family
    phase=torch.tensor([False]);false=torch.tensor([False])
    for distance,expected in ((1.2,False),(.69,True),(.8,True),(.96,False),(.8,False)):
        phase=door_open_amp_phase(phase,torch.tensor([distance]),false,false,.7,.95)
        assert phase.item()==expected
    assert door_open_amp_phase(false,torch.tensor([1.2]),torch.tensor([True]),false,.7,.95).item()
    tasks=torch.tensor([0,0,1,1]);hand=torch.tensor([0,0,0,1]);hold=torch.tensor([False,False,False,True]);opening=torch.tensor([False,False,True,True])
    assert door_amp_family(tasks,hand,hold,opening,6,pushing=torch.tensor([False,True,False,False])).tolist()==[5,0,1,4]
    assert door_amp_family(tasks,hand,hold,opening,6).tolist()==[0,0,1,4]


def test_push_stability_penalties_reject_tilt_and_lift():
    from utils.push_door_spec import push_stability_penalties
    import math
    cfg=dict(max_tilt_degrees=15.,tilt_free_degrees=5.,tilt_penalty_weight=.2,lift_penalty_weight=.2,lift_penalty_scale=.04)
    tilt,lift=push_stability_penalties(torch.tensor([1.,math.cos(math.radians(10)),math.cos(math.radians(20))]),torch.tensor([0.,.02,.08]),cfg)
    torch.testing.assert_close(tilt,torch.tensor([0.,-.1,-.2]),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(lift,torch.tensor([0.,-.1,-.2]))


def test_low_push_hand_band_and_ground_contact_slack():
    from utils.push_door_spec import push_hand_alignment,push_stability_penalties
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi_variable_box.yaml').read_text())['env']['interaction']['push']
    torch.testing.assert_close(torch.tensor(cfg['hand_height_fraction'])*1.3,torch.tensor([.7,.9]))
    hands=torch.tensor([[[-.69,-.2,.15],[-.69,.2,.15]]])
    direction=torch.tensor([[1.,0.,0.]])
    good,_=push_hand_alignment(hands,direction,torch.tensor([.65]*3),cfg)
    raised=hands.clone();raised[...,2]=.45
    bad,_=push_hand_alignment(raised,direction,torch.tensor([.65]*3),cfg)
    assert good>.99 and bad<.2
    _,lift=push_stability_penalties(torch.ones(3),torch.tensor([.005,.01,.04]),cfg)
    torch.testing.assert_close(lift,torch.tensor([0.,0.,-.2]))


def test_unconditioned_amp_keeps_motion_features_without_labels():
    from utils.push_door_spec import amp_motion_sample
    motion=torch.randn(12,129)
    families=torch.arange(12)%6
    mixed=amp_motion_sample(motion,families,6,False)
    torch.testing.assert_close(mixed,motion)
    assert mixed.shape==(12,129)
    conditional=amp_motion_sample(motion,families,6)
    torch.testing.assert_close(conditional[:,:129],motion)
    torch.testing.assert_close(conditional[:,129:],torch.nn.functional.one_hot(families,6).float())


def test_mixed_amp_snapshot_changes_only_amp_conditioning():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    baseline=yaml.safe_load((folder/'push_door_stage1_box13_10500_baseline.yaml').read_text())
    mixed=yaml.safe_load((folder/'push_door_stage1_box13_10500_mixed_amp.yaml').read_text())
    assert mixed['env']['interaction']['amp'].pop('task_conditioning') is False
    assert mixed==baseline
    validate_interaction(baseline['env']['interaction'])
    baseline['env']['interaction']['amp']['task_conditioning']=False
    validate_interaction(baseline['env']['interaction'])
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_box13_10500_mixed_amp.yaml').read_text())
    assert cfg['params']['load_checkpoint'] is False
    assert cfg['params']['seed']==9324


def test_mixed_demo_sampler_preserves_sources_and_temporal_history(monkeypatch):
    from types import SimpleNamespace
    import env.tasks.multi_agent.humanoid_ma_push_door as module
    task=object.__new__(module.HumanoidMAPushDoor)
    task.device='cpu';task.dt=1/30;task._amp_label_count=6
    task._num_amp_obs_steps=10;task._num_amp_obs_per_step=129
    task._amp_task_conditioning=False
    task._interaction={'amp':{'push_phase':[0.,1.],'open_phase':[0.,.7],
                             'hold_phase':[.7,1.],'hold_source':'door_tail'}}
    task._door_reference_hands=torch.tensor([0,1])
    task._local_root_obs=True;task._root_height_obs=True
    task._dof_obs_size=0;task._dof_offsets=[]
    calls=[]
    def library(name):
        def state(ids,times):
            calls.append((name,ids.clone(),times.clone()))
            obs=torch.zeros(len(ids),129);obs[:,0]=times;obs[:,1]=ids
            return (obs,)*7
        return SimpleNamespace(_motion_weights=torch.ones(2)/2,
            sample_motions=lambda count:torch.randint(2,(count,)),
            get_motion_length=lambda ids:torch.full_like(ids,4,dtype=torch.float),
            get_motion_state=state)
    task._motion_lib={name:library(name) for name in ('push','doorOpen','loco')}
    monkeypatch.setattr(module,'build_amp_observations',lambda root,*args:root)
    torch.manual_seed(19)
    mixed=task.fetch_amp_obs_demo(12)
    assert mixed.shape==(12,1290) and torch.isfinite(mixed).all()
    assert [c[0] for c in calls]==['push','doorOpen','doorOpen','doorOpen','doorOpen','loco']
    for family,(name,ids,times) in enumerate(calls):
        ids=ids.view(2,10);times=times.view(2,10)
        assert torch.all(ids==ids[:,:1])
        torch.testing.assert_close(times[:,:-1]-times[:,1:],torch.full((2,9),task.dt))
        lo,hi=task._expert_source(family)[1]
        assert torch.all(times>=4*lo) and torch.all(times<=4*hi)
        if 1<=family<=4:assert torch.all(ids==(family-1)%2)
    task._amp_task_conditioning=True;task._num_amp_obs_per_step=135
    torch.manual_seed(19)
    conditional=task.fetch_amp_obs_demo(12).view(12,10,135)
    torch.testing.assert_close(conditional[:,:,:129],mixed.view(12,10,129))


def test_unified_door_bias_is_identical_for_open_and_hold():
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    opening=task_relation_matrix(tasks,torch.zeros_like(tasks,dtype=torch.bool),True,True)
    holding=task_relation_matrix(tasks,torch.ones_like(tasks,dtype=torch.bool),True,True)
    torch.testing.assert_close(opening,holding)
    for row in range(4):
        for owner in range(2):
            obj=2+owner+2*tasks[row,owner]
            relation=12 if tasks[row,owner] else 11
            assert opening[row,owner,obj]==relation and opening[row,obj,owner]==relation
            assert opening[row,obj,6+owner]==(12 if tasks[row,owner] else 7)
            assert opening[row,6+owner,obj]==opening[row,obj,6+owner]
    assert not (opening==13).any()
    legacy=task_relation_matrix(tasks,torch.ones_like(tasks,dtype=torch.bool),True)
    assert (legacy==13).any()
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    for name in ('push_door_stage1','push_door_stage1_random_start','push_door_stage1_task_rsi',
                 'push_door_stage1_task_rsi_variable_box','push_door_stage1_semantic_25kg_mixed_amp'):
        env=yaml.safe_load((folder/(name+'.yaml')).read_text())['env']
        assert env['relationGraph']['unified_door_relation'] is True


def test_stage2_dense_push_rewards_reach_distant_agents_without_contact():
    from utils.push_door_spec import stage2_contact_at_rewards
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_semantic_25kg_mixed_amp_dense.yaml').read_text())['env']['interaction']['push']['relation_rewards']
    quality=torch.zeros(2);no=torch.zeros(2,dtype=torch.bool)
    far=stage2_contact_at_rewards(quality,no,torch.tensor([2.,1.]),torch.tensor([4.,2.]),no,cfg)
    torch.testing.assert_close(far[:,1],torch.tensor([.08,.2/1.5]))
    torch.testing.assert_close(far[:,4],torch.tensor([.2/4.5,.08]))
    assert far[1,1]>far[0,1] and far[1,4]>far[0,4]
    repeated=stage2_contact_at_rewards(quality,~no,torch.tensor([2.,1.]),torch.tensor([4.,2.]),no,cfg)
    torch.testing.assert_close(repeated[:,:3],torch.full((2,3),.2))
    torch.testing.assert_close(repeated[:,4],far[:,4])
    finished=stage2_contact_at_rewards(quality,no,torch.tensor([2.,1.]),torch.tensor([0.,0.]),~no,cfg)
    torch.testing.assert_close(finished,torch.full((2,6),.2))


def test_stage2_push_xy_quality_is_broad_and_requires_both_hands():
    from utils.push_door_spec import stage2_push_hand_quality
    config={'hand_near_scale':10.}
    direction=torch.tensor([[1.,0.,0.]])
    half=torch.tensor([[.65,.65,.65]])
    hands=torch.tensor([[[-.65,-.2,.1],[-.65,.2,.4]]])
    torch.testing.assert_close(stage2_push_hand_quality(hands,direction,half,config),torch.ones(1))
    far=hands.clone();far[:,1,0]-=.5
    phi=stage2_push_hand_quality(far,direction,half,config)
    torch.testing.assert_close(phi,torch.exp(torch.tensor([-2.5])))
    assert phi>.08
    opposite=hands.clone();opposite[...,0]=.65
    assert stage2_push_hand_quality(opposite,direction,half,config)<1e-6


def test_dense_push_experiment_only_changes_reward_configuration():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_mixed_amp.yaml').read_text())
    new=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_mixed_amp_dense.yaml').read_text())
    validate_interaction(new['env']['interaction'])
    new['env']['interaction']['push']['relation_rewards']=old['env']['interaction']['push']['relation_rewards']
    assert new==old


@pytest.mark.parametrize('push_mix,push_fraction', [([.33,.67,0.],.67),([0.,1.,0.],1.)])
def test_task_conditioned_mixture_never_crosses_push_and_door(monkeypatch,push_mix,push_fraction):
    from types import SimpleNamespace
    import env.tasks.multi_agent.humanoid_ma_push_door as module
    task=object.__new__(module.HumanoidMAPushDoor)
    task.device='cpu';task.dt=1/30;task._amp_label_count=2;task._amp_source_count=6
    task._num_amp_obs_steps=10;task._num_amp_obs_per_step=131;task._amp_task_conditioning=True
    task._amp_expert_mix={'push':push_mix,'door':[.33,0.,.67]}
    task._local_root_obs=True;task._root_height_obs=True;task._dof_obs_size=0;task._dof_offsets=[]
    def library(code):
        def state(ids,times):
            obs=torch.zeros(len(ids),129);obs[:,0]=code;obs[:,1]=times
            return (obs,)*7
        return SimpleNamespace(sample_motions=lambda count:torch.zeros(count,dtype=torch.long),
            get_motion_length=lambda ids:torch.full_like(ids,4,dtype=torch.float),get_motion_state=state)
    task._motion_lib={name:library(code) for code,name in enumerate(('loco','push','doorOpen'))}
    monkeypatch.setattr(module,'build_amp_observations',lambda root,*args:root)
    torch.manual_seed(42)
    demos=task.fetch_amp_obs_demo(4096).view(4096,10,131)
    labels=demos[:,:,-2:].argmax(-1)
    assert torch.all(labels==labels[:,:1])
    codes=demos[:,:,0];assert torch.all(codes==codes[:,:1])
    push=labels[:,0]==0;door=~push
    assert not (codes[push]==2).any() and not (codes[door]==1).any()
    for selected,code,expected in ((push,1,push_fraction),(door,2,.67)):
        assert abs(float((codes[selected,0]==code).float().mean())-expected)<.04
    if push_fraction==1.:assert torch.all(codes[push]==1)
    torch.testing.assert_close(demos[:,:-1,1]-demos[:,1:,1],torch.full((4096,9),task.dt),atol=1e-6,rtol=1e-5)
    task._tasks=torch.tensor([[0,1],[1,0]])
    # Neither approaching nor holding changes task labels.
    task._holding_phase=torch.ones(2,2,dtype=torch.bool)
    torch.testing.assert_close(task._family(),task._tasks)
    torch.testing.assert_close(task._family(torch.tensor([1])),task._tasks[1:])


def test_task_mix_config_changes_only_amp_distribution():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_conditional_amp_dense.yaml').read_text())
    new=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_task_mix_amp_dense.yaml').read_text())
    validate_interaction(new['env']['interaction'])
    bad=copy.deepcopy(new);bad['env']['interaction']['amp']['task_expert_mix']['push']=[.33,.33,.34]
    with pytest.raises(ValueError):validate_interaction(bad['env']['interaction'])
    assert new['env']['interaction']['amp'].pop('task_expert_mix')=={'push':[.33,.67,0.],'door':[.33,0.,.67]}
    assert new==old


def test_door_goal_split_changes_only_door_goal_edges_and_preserves_old_mapping():
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    for holding in (torch.zeros_like(tasks,dtype=torch.bool),torch.ones_like(tasks,dtype=torch.bool)):
        old=task_relation_matrix(tasks,holding,True,True)
        new=task_relation_matrix(tasks,holding,True,True,True)
        changed=torch.zeros_like(old,dtype=torch.bool)
        for row in range(4):
            for owner in range(2):
                if tasks[row,owner]:
                    door=4+owner;goal=6+owner
                    changed[row,door,goal]=changed[row,goal,door]=True
                    assert new[row,owner,door]==12 and new[row,door,owner]==12
                    assert new[row,door,goal]==5 and new[row,goal,door]==5
        assert torch.equal(new!=old,changed)
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    base=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_legacy_amp_rsi_dense.yaml').read_text())
    split=yaml.safe_load((folder/'push_door_stage1_semantic_25kg_legacy_amp_rsi_dense_split_goal.yaml').read_text())
    assert split['env']['relationGraph'].pop('generic_door_goal') is True
    assert split==base
    changed=copy.deepcopy(base);changed['env']['relationGraph']['generic_door_goal']=True
    with pytest.raises(ValueError):
        check_interaction_checkpoint({'interaction_metadata':interaction_metadata(base['env'])},interaction_metadata(changed['env']))


def test_split_door_goal_ids_reach_attention_encoder():
    import torch.nn as nn
    from learning.multi_agent.amp_network_builder_ma import RelationEncoder
    encoder=RelationEncoder([223,34,9],2,4,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias=True,
        relation_bias_mode='edge_mlp',relation_graph_spec={'template':'push_door_stage1_v1',
            'semantic_relations':True,'unified_door_relation':True,'generic_door_goal':True},
        gta_cfg={'enable':True,'translation_scale':1.,'representation':'se3_direct_sum'})
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    obs=torch.randn(4,656);goals=obs[:,582:600].view(4,2,9)
    goals[...,:2]=torch.nn.functional.one_hot(tasks,2).float()
    obs[:,600:].view(4,8,7)[...,3:]=torch.tensor([0.,0.,0.,1.])
    seen=[]
    hook=encoder.edge_encoder.register_forward_pre_hook(lambda module,args:seen.append(args[1].clone()))
    with torch.no_grad():encoder.edge_encoder.bias_projection.normal_(std=.02)
    encoder(obs).square().sum().backward();hook.remove()
    torch.testing.assert_close(seen[-1],task_relation_matrix(tasks,semantic=True,unified_door=True,generic_door_goal=True))
    grad=encoder.edge_encoder.relation_embed.weight.grad
    assert grad[5].abs().sum()>0 and grad[12].abs().sum()>0


def test_old_arch_experiment_keeps_current_push_and_door_rewards():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    before=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_semantic_25kg_legacy_amp_rsi_dense_split_goal_old_spacing.yaml').read_text())
    after=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_dense.yaml').read_text())
    original=yaml.safe_load((folder/'push_door_stage1_box13_10500_baseline.yaml').read_text())
    assert after['env']['relationGraph']==original['env']['relationGraph']
    assert after['env']['interaction']==before['env']['interaction']
    after['env']['relationGraph']=before['env']['relationGraph']
    assert after==before


def door_spr_inputs(n=1):
    import math
    cfg=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr.yaml').read_text())['env']['interaction']['door']
    hands=torch.tensor([[[.02,0.,0.],[3.,0.,0.]]]).expand(n,-1,-1).clone()
    handles=torch.tensor([[[0.,0.,0.],[4.,0.,0.]]]).expand(n,-1,-1).clone()
    forces=torch.tensor([[1.,0.]]).expand(n,-1).clone()
    return cfg,hands,handles,forces,torch.zeros(n)


def test_door_spr_approach_dense_and_exact_contact_state_requires_same_pair():
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(3)
    h[:,0,0]=torch.tensor([.05,.5,1.])
    r=door_contact_open_rewards(h,g,z[:,None].expand(-1,2),f,z,z,z,z,.1,c)
    progress=r['components'][:,1]
    assert progress[0]>progress[1]>progress[2]>0
    assert not r['contact'].any() and torch.all(r['components'][:,0]==0)
    c,h,g,f,z=door_spr_inputs()
    mismatch=torch.tensor([[0.,1.]])
    bad=door_contact_open_rewards(h,g,mismatch,f,z,z,z,z,.1,c)
    assert not bad['contact'].item()
    good=door_contact_open_rewards(h,g,f,f,z,z,z,z,.1,c)
    assert good['contact'].item() and good['components'][0,0]>0


def test_door_spr_angle_progress_requires_contact_and_decreases_when_closing():
    import math
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(3)
    angles=torch.deg2rad(torch.tensor([0.,40.,80.]))
    r=door_contact_open_rewards(h,g,f,f,angles,z,z,z,.1,c)
    torch.testing.assert_close(r['components'][:,4],torch.tensor([0.,.1,.2]))
    torch.testing.assert_close(r['components'][:,3],torch.full((3,),.2))
    lost=door_contact_open_rewards(h,g,torch.zeros_like(f),f,angles,z,z,z,.1,c)
    assert torch.all(lost['components'][:,3:]==0)


def test_door_spr_continuous_hold_recurring_success_and_release_reset():
    import math
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs();contact_time=z.clone();open_time=z.clone()
    angle=torch.full_like(z,math.radians(85))
    for step in range(21):
        r=door_contact_open_rewards(h,g,f,f,angle,z,contact_time,open_time,.1,c)
        contact_time=r['contact_elapsed'];open_time=r['open_elapsed']
        if step==0:assert not r['contact_success'].item()
        if step==1:assert r['contact_success'].item()
        if step<18:assert not r['open_success'].item()
    assert r['open_success'].item()
    torch.testing.assert_close(r['components'],torch.full((1,6),.2))
    lost=door_contact_open_rewards(h,g,torch.zeros_like(f),f,angle,z,contact_time,open_time,.1,c)
    assert not lost['open_success'].item() and not lost['contact_success'].item()
    assert lost['open_elapsed'].item()==0 and lost['contact_elapsed'].item()==0
    reacquired=door_contact_open_rewards(h,g,f,f,angle,z,lost['contact_elapsed'],lost['open_elapsed'],.1,c)
    assert not reacquired['open_success'].item() and not reacquired['contact_success'].item()


def test_door_spr_hold_rejects_fast_or_under_angle_and_preserves_push_config():
    import math
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(2)
    angles=torch.deg2rad(torch.tensor([79.,85.]))
    result=door_contact_open_rewards(h,g,f,f,angles,torch.tensor([0.,.31]),z,torch.ones(2),.1,c)
    assert torch.all(result['open_elapsed']==0) and not result['open_success'].any()
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_dense.yaml').read_text())
    new=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr.yaml').read_text())
    validate_interaction(new['env']['interaction'])
    bad=copy.deepcopy(new);bad['env']['interaction']['door']['relation_rewards']['approach_sigma']=0
    with pytest.raises(ValueError):validate_interaction(bad['env']['interaction'])
    with pytest.raises(ValueError):check_interaction_checkpoint({'interaction_metadata':interaction_metadata(old['env'])},interaction_metadata(new['env']))
    new['env']['interaction']['door']=old['env']['interaction']['door']
    assert new==old


def test_door_spr_walk_open_only_changes_amp_and_keeps_push_reference_pure():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    before=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr.yaml').read_text())
    after=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr_walk_open.yaml').read_text())
    validate_interaction(after['env']['interaction'])
    amp=after['env']['interaction']['amp']
    assert amp.pop('task_conditioning') is True
    assert amp.pop('task_expert_mix')=={'push':[0.,1.,0.],'door':[.33,0.,.67]}
    assert after==before


def test_door_contact_score_success_is_immediate_and_not_duration_based():
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(2)
    c['relation_rewards'].pop('contact_hold_seconds')
    c['relation_rewards']['contact_success_threshold']=.9
    h[:,0,0]=torch.tensor([.02,.04])
    r=door_contact_open_rewards(h,g,f,f,z,z,torch.full_like(z,100),z,.001,c)
    assert r['contact_success'].tolist()==[True,False]
    assert torch.all(r['contact_elapsed']==0)
    torch.testing.assert_close(r['components'][:,2],torch.tensor([.2,0.]))
    lost=door_contact_open_rewards(h,g,torch.zeros_like(f),f,z,z,z,z,.1,c)
    assert not lost['contact_success'].any()
    regained=door_contact_open_rewards(h,g,f,f,z,z,z,z,.001,c)
    assert regained['contact_success'][0]


def test_door_score_open_success_cannot_override_low_hand_score():
    import math
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs()
    c['relation_rewards'].pop('contact_hold_seconds');c['relation_rewards']['contact_success_threshold']=.9
    h[:,0,0]=.07
    r=door_contact_open_rewards(h,g,f,f,torch.full_like(z,math.radians(85)),z,z,torch.full_like(z,2),.1,c)
    assert r['open_success'].item() and not r['contact_success'].item()
    assert r['components'][0,2]==0 and r['components'][0,5]==.2


def test_door_score_config_changes_only_contact_success_rule():
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    before=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr_walk_open.yaml').read_text())
    after=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr_walk_open_score.yaml').read_text())
    validate_interaction(after['env']['interaction'])
    invalid=copy.deepcopy(after);invalid['env']['interaction']['door']['relation_rewards']['contact_success_threshold']=1.1
    with pytest.raises(ValueError):validate_interaction(invalid['env']['interaction'])
    relation=after['env']['interaction']['door']['relation_rewards']
    assert relation.pop('contact_success_threshold')==.9
    relation['contact_hold_seconds']=.2
    assert after==before


def test_final_door_spr_experiment_uses_split_bias_without_changing_reward_amp():
    import torch.nn as nn
    from learning.multi_agent.amp_network_builder_ma import RelationEncoder
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    before=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_old_arch_25kg_door_spr_walk_open_score.yaml').read_text())
    after=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_split_bias_25kg_door_spr_walk_open_score.yaml').read_text())
    graph=after['env']['relationGraph']
    assert all(graph.get(k) is True for k in ('semantic_relations','unified_door_relation','generic_door_goal'))
    encoder=RelationEncoder([223,34,9],2,4,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias=True,
        relation_bias_mode='edge_mlp',relation_graph_spec=graph,
        gta_cfg={'enable':True,'translation_scale':1.,'representation':'se3_direct_sum'})
    assert encoder.edge_encoder.relation_embed.num_embeddings==14
    after['env']['relationGraph']=before['env']['relationGraph']
    assert after==before


def test_hand_open_relations_bind_only_correct_door_pairs():
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    for held in (torch.zeros_like(tasks,dtype=torch.bool),torch.ones_like(tasks,dtype=torch.bool)):
        matrix=task_relation_matrix(tasks,held,semantic=True,hand_open=True)
        for row in range(4):
            for owner in range(2):
                obj=2+owner+2*tasks[row,owner];goal=6+owner
                assert matrix[row,owner,obj]==(14 if tasks[row,owner] else 11)
                assert matrix[row,obj,owner]==matrix[row,owner,obj]
                assert matrix[row,obj,goal]==(15 if tasks[row,owner] else 7)
                assert matrix[row,goal,obj]==matrix[row,obj,goal]
                assert matrix[row,owner,goal]==4
        assert not (matrix==5).any() and not (matrix==12).any() and not (matrix==13).any()


def test_hand_open_config_reaches_attention_and_preserves_reward_amp():
    import torch.nn as nn
    from learning.multi_agent.amp_network_builder_ma import RelationEncoder
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((ROOT/'tokenhsi/tests/fixtures/push_door/push_door_stage1_split_bias_25kg_door_spr_walk_open_score.yaml').read_text())
    new=yaml.safe_load((folder/'push_door_stage1_hand_open_25kg_spr_walk_open_score.yaml').read_text())
    graph=new['env']['relationGraph']
    encoder=RelationEncoder([223,34,9],2,4,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias=True,
        relation_bias_mode='edge_mlp',relation_graph_spec=graph,
        gta_cfg={'enable':True,'translation_scale':1.,'representation':'se3_direct_sum'})
    assert encoder.edge_encoder.relation_embed.num_embeddings==16
    tasks=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    obs=torch.randn(4,656);obs[:,582:600].view(4,2,9)[...,:2]=torch.nn.functional.one_hot(tasks,2).float()
    obs[:,600:].view(4,8,7)[...,3:]=torch.tensor([0.,0.,0.,1.])
    seen=[];hook=encoder.edge_encoder.register_forward_pre_hook(lambda module,args:seen.append(args[1].clone()))
    with torch.no_grad():encoder.edge_encoder.bias_projection.normal_(std=.02)
    encoder(obs).square().sum().backward();hook.remove()
    torch.testing.assert_close(seen[-1],task_relation_matrix(tasks,semantic=True,hand_open=True))
    gradient=encoder.edge_encoder.relation_embed.weight.grad
    assert all(gradient[i].abs().sum()>0 for i in (7,11,14,15))
    assert interaction_metadata(new['env'])['relation_taxonomy']['HAND']==14
    assert interaction_metadata(new['env'])['relation_taxonomy']['OPEN']==15
    assert new['env']['interaction']['door']['relation_rewards'].pop('open_state_mode')=='distance'
    assert new['env']['interaction']['door']['relation_rewards'].pop('progress_reference')=='body_door_xy'
    assert new['env']['interaction']['door']['relation_rewards']['open_state_weight']==0.2
    new['env']['interaction']['door']['relation_rewards']['open_state_weight']=old['env']['interaction']['door']['relation_rewards']['open_state_weight']
    assert 'task_expert_mix' not in new['env']['interaction']['amp']
    new['env']['interaction']['amp']['task_expert_mix']=old['env']['interaction']['amp']['task_expert_mix']
    new['env']['relationGraph']=old['env']['relationGraph']
    assert new==old


def test_current_door_taxonomy_removes_old_names_without_breaking_checkpoint_ids():
    import utils.edge_interaction_spec as relations
    assert not hasattr(relations,'DOOR_OPEN') and not hasattr(relations,'DOOR_HOLD')
    assert relations.HAND==14 and relations.OPEN==15
    path=ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_hand_open_25kg_spr_walk_open_score.yaml'
    env=yaml.safe_load(path.read_text())['env'];metadata=interaction_metadata(env)
    assert metadata['relation_taxonomy']=={'AT':7,'PUSH':11,'HAND':14,'OPEN':15}
    historical=copy.deepcopy(metadata);historical['relation_taxonomy'].update(DOOR_OPEN=12,DOOR_HOLD=13)
    check_interaction_checkpoint({'interaction_metadata':historical},metadata)
    wrong=copy.deepcopy(historical);wrong['relation_taxonomy']['HAND']=12
    with pytest.raises(ValueError):check_interaction_checkpoint({'interaction_metadata':wrong},metadata)


def test_door_body_progress_uses_xy_and_ignores_hand_noise():
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(3)
    c['relation_rewards']['progress_reference']='body_door_xy'
    body=torch.tensor([[.5,0.,0.],[.5,0.,2.],[2.,0.,0.]])
    door=torch.zeros_like(body)
    h[:,0,2]=torch.tensor([0.,1.,1.])
    r=door_contact_open_rewards(h,g,f,f,z,z,z,z,.01,c,body,door)
    assert torch.allclose(r['components'][0,1],r['components'][1,1])
    assert r['components'][1,1]>r['components'][2,1]
    assert r['contact'].tolist()==[True,False,False]
    assert r['components'][0,0]>0 and r['components'][1,0]==0
    with pytest.raises(ValueError):
        door_contact_open_rewards(h,g,f,f,z,z,z,z,.01,c)


def test_current_door_open_state_is_soft_distance_without_contact_gate():
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs(3)
    path=ROOT/'tokenhsi/data/cfg/multi_agent/push_door_stage1_hand_open_25kg_spr_walk_open_score.yaml'
    c=yaml.safe_load(path.read_text())['env']['interaction']['door']
    h[:,0,0]=torch.tensor([0.,.05,.1])
    pos=torch.zeros(3,3)
    r=door_contact_open_rewards(h,g,torch.zeros_like(f),f,z,z,z,z,.1,c,pos,pos)
    expected=.2*torch.exp(-100*torch.tensor([0.,.05,.1]).square())
    torch.testing.assert_close(r['components'][:,3],expected)
    assert not r['contact'].any()
    assert (r['components'][:,4:]==0).all()
    assert r['components'][0,3]>r['components'][1,3]>r['components'][2,3]>0
    h[:,0,2]=.1
    lower=door_contact_open_rewards(h,g,torch.zeros_like(f),f,z,z,z,z,.1,c,pos,pos)
    assert (lower['components'][:,3]<r['components'][:,3]).all()


def test_right_hand_door_filters_data_and_keeps_push_amp_separate():
    from utils.push_door_spec import door_motion_matches,door_motion_hand
    folder=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((folder/'push_door_stage1_hand_open_25kg_spr_walk_open_score.yaml').read_text())
    new=yaml.safe_load((folder/'push_door_stage1_right_hand_25kg_spr_task_mix.yaml').read_text())
    interaction=new['env']['interaction'];validate_interaction(interaction)
    assert interaction['task_rsi'].pop('door_alignment_yaw_degrees')==[-45.,-22.5,0.,22.5,45.]
    assert interaction['door'].pop('working_hand')=='right'
    assert interaction['amp'].pop('task_expert_mix')=={'push':[0.,1.,0.],'door':[.33,0.,.67]}
    assert interaction['amp']['door_hand_selection']=='fixed_right_v1'
    interaction['amp']['door_hand_selection']=old['env']['interaction']['amp']['door_hand_selection']
    assert new==old
    dataset=yaml.safe_load((ROOT/'tokenhsi/data/dataset_push_door_stage1.yaml').read_text())
    # Check actual dataset entries rather than hand labels inferred from filenames alone.
    entries=dataset['motions']['doorOpen']
    selected=[v['file'] for v in entries if v['weight']>0 and door_motion_matches(v['file'],'left_open','right')]
    assert selected and all(door_motion_hand(p)==0 for p in selected)
    assert not door_motion_matches('motions/inside_door_handle_right_side_open_walk_R_001__A512_M/phys_humanoid_v3/ref_motion.npy','left_open','right')


def test_right_hand_door_rewards_reject_left_hand_only_contact():
    from utils.push_door_spec import door_contact_open_rewards
    c,h,g,f,z=door_spr_inputs()
    h[:,0]=torch.tensor([2.,0.,0.]);h[:,1]=torch.tensor([.02,0.,0.])
    f=torch.ones_like(f)
    any_hand=door_contact_open_rewards(h,g,f,f,z,z,z,z,.01,c)
    assert any_hand['contact'].item()
    c['working_hand']='right'
    right_only=door_contact_open_rewards(h,g,f,f,z,z,z,z,.01,c)
    assert not right_only['contact'].item() and right_only['components'][0,0]==0
    h[:,0]=torch.tensor([.02,0.,0.])
    right=door_contact_open_rewards(h,g,f,f,z,z,z,z,.01,c)
    assert right['contact'].item() and right['components'][0,0]>0
