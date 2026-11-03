"""Solo PUSH must preserve BEFORE pairs, transfer identity and physical isolation."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from utils.experiments import native_config
from utils.before_push_spec import sample_graph,validate_env
from utils.before_spec import pack_objects,task_objects
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.human_coordination import human_relations
from learning.multi_agent.before_push_network import BeforePushBuilder,transfer_stage1
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime

ENV=native_config('stage2_before_push')['env']
RANGES=ENV['box']['build']['taskSizeRanges']
PROBS=ENV['relationGraph']['template_probabilities']


def draw(n=256,preset='random_scenario',sizes=None):
    return sample_graph(torch.full((n,4,3),.4) if sizes is None else sizes,RANGES,PROBS,preset)


def test_solo_prior_pair_coin_and_eligible_family_uniformity():
    torch.set_num_threads(1);torch.manual_seed(65)
    g,pre,t=draw(40000)
    paired=pre>=0;r,f=paired.nonzero(as_tuple=True);leaders=pre[r,f]
    paired[r,leaders]=True
    frequency=torch.bincount(t[~paired],minlength=6)[1:]/(~paired).sum()
    torch.testing.assert_close(frequency,torch.tensor([.1,.15,.25,.2,.3]),atol=.009,rtol=0)
    torch.testing.assert_close(torch.bincount((pre>=0).sum(-1),minlength=3)/40000.,torch.tensor([.008,.352,.64]),atol=.01,rtol=0)
    assert (t[r,leaders]==3).all() and torch.isin(t[r,f],torch.tensor([1,2,4])).all()
    family_counts=torch.bincount(t[r,f],minlength=6)[[1,2,4]]/len(r)
    torch.testing.assert_close(family_counts,torch.full((3,),1/3),atol=.009,rtol=0)
    relations=human_relations(task_graph_packet(g,len(t)),4)
    assert (relations[r,f,leaders]==2).all()
    assert int((relations==2).sum())==len(r)
    assert not ((t==5)&paired).any()


@pytest.mark.parametrize('preset',['push','all_solo','before_sit','before_climb','before_stack','one_pair_stack'])
def test_templates_and_physical_mapping(preset):
    g,pre,t=draw(preset=preset)
    if preset=='push':assert (t==5).all() and (pre==-1).all()
    people=torch.arange(4)[None].expand(len(t),-1)
    primary=people+torch.where(t==5,16,torch.where(t==1,4,torch.where(t==2,8,0)))
    parents=torch.cat((primary,people+12),-1)
    valid=task_objects(pre,t)
    mapping,inverse,count=pack_objects(parents,valid,8,20,(16,20))
    for r in range(len(t)):
        assert len(mapping[r].unique())==8
        assert (mapping[r,count[r]:]<16).all()
        assert torch.equal(mapping[r,inverse[r,valid[r]]],parents[r,valid[r]])
    assert (g.prereq_mask.sum((1,2))==(pre>=0).sum(-1)).all()


def test_contact_weights_and_before_gate_are_independent():
    g,pre,t=draw(n=1,preset='one_pair_stack')
    # Force a free person to PUSH without changing the paired records.
    paired=pre>=0;r,f=paired.nonzero(as_tuple=True);paired[r,pre[r,f]]=True
    free=(~paired).nonzero(as_tuple=True)[1][0]
    t[0,free]=5
    from utils.stage1_jointpush import compose_graph
    push_graph=compose_graph(t,num_objects=8)
    contact=(g.edge_owner==free)&(push_graph.edge_relation==12)
    g.edge_relation[contact]=12
    runtime=BeforePushRuntime(1,g,ENV['relationReward'],'cpu',contact_weights=ENV['jointPush']['contact_weights'])
    phi=torch.ones(1,len(g.ids));zero=torch.zeros_like(phi)
    runtime.reset(torch.tensor([0]),phi,zero,zero,zero)
    result=runtime.step(phi,phi,zero,zero,zero)
    assert not result['own_success'][contact].any()
    # AT success saturates its paired contact; a failed prerequisite still gates the follower.
    torch.testing.assert_close(result['state_component'][contact],torch.full_like(phi[contact],.2))
    leader_edge=g.prereq_mask.any(1)
    phi[leader_edge]=0
    follower_edge=g.prereq_mask.any(-1)
    result=runtime.step(phi,phi,zero,zero,zero)
    assert not result['prerequisite_gate'][follower_edge].any()
    assert not result['own_success'][follower_edge].any()
    runtime.contact[0,free]=True
    assert runtime.step(phi,phi,zero,zero,zero)['own_success'][contact].all()
    runtime.reset(torch.tensor([0]),phi,zero,zero,zero)
    assert not runtime.contact.any()


def test_actual_push_transfer_copies_every_tensor_and_freezes_only_actor_encoder():
    path=Path('checkpoints/Stage1PushFinetune_00021000.pth')
    if not path.is_file():pytest.skip('Connect the Stage1 PUSH checkpoint')
    builder=BeforePushBuilder();builder.load(native_config('stage2_before_push',True)['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(1272,),amp_input_shape=(1330,),value_size=1,num_agents=4,num_objects=8,
        observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=16.,
        relation_reward_mode=ENV['relationReward']['mode'],relation_graph_spec=ENV['relationGraph'],device='cpu')
    model=torch.nn.Module();model.a2c_network=net
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    report=transfer_stage1(model,checkpoint)
    assert report['source_epoch']==21000 and set(report['copied'])==set(checkpoint['model'])
    for key,value in checkpoint['model'].items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert all(not p.requires_grad for p in net.actor_encoder.parameters())
    assert all(p.requires_grad for p in net.action_head.parameters())
    assert all(p.requires_grad for p in net.coordination.parameters())
    with pytest.raises(ValueError,match='Stage1 PUSH'):
        transfer_stage1(model,{'model':{},'relation_experiment_config':{'env':native_config('stage1')['env']}})


def test_contract_rejects_pair_or_solo_drift():
    validate_env(ENV)
    for key,value in [('numAgents',0),('before',{}),('templateRsi',{'PUSH_AT':[0.]*8+[1.]})]:
        cfg=deepcopy(ENV);cfg[key]=value
        with pytest.raises(ValueError):validate_env(cfg)


def test_push_corridors_with_four_humans_and_twenty_physical_objects():
    from utils.before_push_layout import sample_layout,paths_valid
    from utils.jointpush_scene import goal_angles_valid
    torch.manual_seed(101)
    n=12;m=4;p=20
    boxes=torch.zeros(n,p,13);boxes[...,6]=1;boxes[...,:2]=1000.
    sizes=torch.full((n,p,3),.4);sizes[:,16:]=torch.tensor([1.8,1.8,1.5])
    roots=torch.zeros(n,m,3);roots[:,:,:2]=torch.tensor([[-3.,-3.],[-3.,3.],[3.,-3.],[3.,3.]])
    primary=torch.arange(16,20)[None].expand(n,-1);active=torch.zeros(n,p,dtype=torch.bool);active[:,16:]=True
    push=torch.ones(n,m,dtype=torch.bool)
    result,goals=sample_layout(boxes,sizes,roots,push,primary,active,ENV['jointPush']['scene'])
    assert paths_valid(result,sizes,goals,roots,push,primary,active,ENV['jointPush']['scene']).all()
    assert goal_angles_valid(result[:,16:],goals,push,ENV['jointPush']['scene']).all()
    for distance in ((result[:,16:,:2]-roots[...,:2]).norm(dim=-1),(result[:,16:,:2]-goals[...,:2]).norm(dim=-1)):
        assert ((distance>=2-1e-6)&(distance<=10+1e-6)).all()


def test_extra_distractors_keep_large_bank_excluded():
    parents=torch.arange(8)[None]
    valid=torch.ones(1,8,dtype=torch.bool)
    mapping,_,_=pack_objects(parents,valid,24,28,(16,20))
    assert len(mapping[0].unique())==24
    assert not ((mapping>=16)&(mapping<20)).any()


def test_viewer_horizon_override_preserves_checkpoint_contract():
    from utils.experiments import validate_experiment,check_before_checkpoint
    from utils.before_spec import checkpoint_contract
    shortened=deepcopy(ENV);shortened['episodeLength']=32
    validate_experiment('stage2_before_push',shortened,native_config('stage2_before_push',True),evaluation=True)
    check_before_checkpoint({'before_contract':checkpoint_contract(ENV)},shortened)
    with pytest.raises(ValueError,match='1200'):
        validate_experiment('stage2_before_push',shortened,native_config('stage2_before_push',True))


@pytest.mark.parametrize('other',['stage2_before','stage2_before_rsi433','stage2_before_nonemask_rsi433_near50'])
def test_original_before_and_push_checkpoints_cannot_be_mixed(other):
    from utils.experiments import validate_experiment,check_before_checkpoint
    from utils.before_spec import checkpoint_contract
    old=native_config(other)['env']
    for source,target in ((ENV,old),(old,ENV)):
        with pytest.raises(ValueError):check_before_checkpoint({'before_contract':checkpoint_contract(source)},target)
    with pytest.raises(ValueError):validate_experiment(other,ENV,native_config(other,True))
    with pytest.raises(ValueError):validate_experiment('stage2_before_push',old,native_config('stage2_before_push',True))
