"""Shared Stage2 architecture, relation semantics and coupled physical contracts."""
from copy import deepcopy
import pytest
import torch
from utils.experiments import native_config,validate_experiment,experiment_horizon
from utils.unified_training import validate_unified_env
from utils.coupled_push_spec import (sample_groups,sample_templates,group_sizes,asset_bank,physical_assignment,
    compose_graph,region_states,group_contact_gate,expected_solo_fraction)
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.stage2_push_network import Stage2PushBuilder,transfer_stage1
from learning.multi_agent.human_coordination import stage2_human_relations
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd

NAMES=('stage2_before_push','stage2_coupled_push')


def build(name):
    env=native_config(name)['env'];b=Stage2PushBuilder();b.load(native_config(name,True)['params']['network'])
    return b.build('amp',actions_num=32,input_shape=(1272,),amp_input_shape=(1330,),value_size=1,num_agents=4,num_objects=8,
        observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=16.,
        relation_reward_mode=env['relationReward']['mode'],relation_graph_spec=env['relationGraph'],device='cpu')


def test_both_tasks_have_the_same_model_and_stage1_transfer():
    torch.set_num_threads(1)
    before,coupled=[build(name) for name in NAMES]
    assert {k:v.shape for k,v in before.state_dict().items()}=={k:v.shape for k,v in coupled.state_dict().items()}
    coupled.load_state_dict(before.state_dict(),strict=True)
    ckpt=torch.load('checkpoints/Stage1PushFinetune_00021000.pth',map_location='cpu',weights_only=False)
    for net in (before,coupled):
        model=torch.nn.Module();model.a2c_network=net
        report=transfer_stage1(model,ckpt)
        assert len(report['copied'])==165 and report['source_epoch']==21000
        for key,value in ckpt['model'].items():assert torch.equal(model.state_dict()[key],value)
        assert all(not p.requires_grad for p in net.actor_encoder.parameters())
        assert all(p.requires_grad for p in net.coordination.parameters())
        assert net.coordination.blocks[0].relation_bias.shape==(2,4)
    rms=SceneRunningMeanStd([223,30,1],[4,8,4],normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=24)
    rms.load_state_dict(ckpt['running_mean_std'],strict=True)


@pytest.mark.parametrize('name',NAMES+('stage2_coupled_push_goal5',))
def test_current_contracts_and_time(name):
    env=native_config(name)['env'];validate_unified_env(env);validate_experiment(name,env,native_config(name,True))
    assert env['episodeLength']==experiment_horizon(name)==1200
    assert env['stage2Policy']==native_config(NAMES[0])['env']['stage2Policy']
    assert native_config(name,True)['params']['network']==native_config(NAMES[0],True)['params']['network']
    cfg=deepcopy(env);cfg['stage2Policy']['relations'].pop()
    with pytest.raises(ValueError):validate_unified_env(cfg)


def test_coupled_short_distances_preserve_solo_and_legacy_contracts():
    from utils.coupled_push_spec import checkpoint_contract,check_checkpoint
    env=native_config('stage2_coupled_push_goal5')['env']
    assert env['stage2JointPush']['source_distance']==[2.,10.]
    assert env['stage2JointPush']['goal_distance']==[2.,5.]
    for key in ('source_distance','goal_distance'):
        assert env['jointPush']['scene'][key]==[2.,10.]
    legacy=native_config('stage2_coupled_push')['env']
    assert 'source_distance' not in legacy['stage2JointPush']
    assert 'goal_distance' not in legacy['stage2JointPush']
    validate_unified_env(legacy)
    checkpoint={'coupled_push_contract':checkpoint_contract(legacy)}
    check_checkpoint(checkpoint,legacy)
    with pytest.raises(ValueError):check_checkpoint(checkpoint,env)


def test_goal5_only_changes_coupled_distances_and_transfer():
    original=native_config('stage2_coupled_push')
    short=native_config('stage2_coupled_push_goal5')
    env=deepcopy(short['env'])
    for key in ('source_distance','goal_distance'):
        env['stage2JointPush'].pop(key)
    assert env==original['env'] and short['sim']==original['sim']
    for name,other in (('stage2_coupled_push',short),('stage2_coupled_push_goal5',original)):
        with pytest.raises(ValueError,match='selected experiment'):
            validate_experiment(name,other['env'],native_config(name,True))


def test_goal5_transfers_its_configured_stage1_26000():
    path=native_config('stage2_coupled_push_goal5')['experiment']['transfer']['checkpoint']
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=torch.nn.Module();model.a2c_network=build('stage2_coupled_push_goal5')
    report=transfer_stage1(model,checkpoint)
    assert report['source_epoch']==26000 and len(report['copied'])==165
    for key,value in checkpoint['model'].items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


@pytest.mark.parametrize('bounds',[[5.,2.],[0.,5.],[2.]])
def test_coupled_rejects_invalid_distance_ranges(bounds):
    env=native_config('stage2_coupled_push')['env']
    env['stage2JointPush']['source_distance']=bounds
    with pytest.raises(ValueError,match='distance range'):validate_unified_env(env)


def test_before_and_coupled_relations_are_distinct_in_the_same_packet_format():
    from utils.before_push_spec import sample_graph
    e=native_config(NAMES[0])['env']
    graph,pre,_=sample_graph(torch.full((8,4,3),.4),e['box']['build']['taskSizeRanges'],e['relationGraph']['template_probabilities'],'before_stack')
    groups=torch.where(pre>=0,pre,torch.arange(4)[None])
    relation=stage2_human_relations(task_graph_packet(graph,8),groups)
    row,f=(pre>=0).nonzero(as_tuple=True)
    assert (relation[row,f,pre[row,f]]==2).all() and not (relation==3).any()
    groups=torch.tensor([[0,0,2,2]])
    graph=compose_graph(torch.full((1,4),5),groups,False)
    relation=stage2_human_relations(task_graph_packet(graph,1),groups)
    assert relation.tolist()==[[[1,3,0,0],[3,1,0,0],[0,0,1,3],[0,0,3,1]]]


def test_group_and_solo_sampling_follow_config():
    torch.manual_seed(42);n=50000
    groups=sample_groups(n,4,.8,'cpu');size=group_sizes(groups)
    e=native_config(NAMES[1])['env'];prob=torch.tensor(list(e['relationGraph']['template_probabilities'].values()))
    task=sample_templates(groups,prob)
    assert (task[size>1]==5).all()
    torch.testing.assert_close((size==1).float().mean(),torch.tensor(expected_solo_fraction(4,.8)),atol=.006,rtol=0)
    freq=torch.bincount(task[size==1],minlength=6)/(size==1).sum()
    torch.testing.assert_close(freq,prob,atol=.009,rtol=0)
    assert group_sizes(sample_groups(16,4,.8,'cpu','all_coupled')).eq(4).all()
    assert group_sizes(sample_groups(16,4,.8,'cpu','all_solo')).eq(1).all()


def test_coupled_box_scales_mass_and_splits_into_equal_regions():
    e=native_config(NAMES[1])['env'];sizes,mass=asset_bank(16,e,'cpu')
    torch.testing.assert_close(mass[:,16:],sizes[:,16:].prod(-1)*3)
    for count,slot in ((2,20),(3,22),(4,23)):
        groups=torch.zeros(16,4,dtype=torch.long)
        if count<4:groups[:,count:]=torch.arange(count,4)
        templates=torch.full_like(groups,5)
        assignment=physical_assignment(templates,groups)
        assert (assignment[:,:count]==slot).all()
    state=torch.zeros(1,4,13);state[...,6]=1
    size=torch.tensor([[[7.2,1.8,1.5]]]).expand(1,4,3)
    tokens,cells=region_states(state,size,torch.full((1,4),4),torch.arange(4)[None])
    torch.testing.assert_close(cells[...,0],torch.full((1,4),1.8))
    torch.testing.assert_close(tokens[0,:,0],torch.tensor([-2.7,-.9,.9,2.7]))


def test_group_gate_requires_every_member_then_saturates_at_goal():
    box=torch.zeros(1,4,13);box[...,6]=1
    size=torch.ones(1,4,3)*2
    goals=torch.zeros(1,4,3);goals[...,1]=5
    groups=torch.zeros(1,4,dtype=torch.long)
    hands=torch.zeros(1,4,2,3);hands[...,1]=-1
    assert group_contact_gate(hands,box,size,goals,groups,goal_saturation=True).all()
    hands[:,0,:,1]=0
    assert not group_contact_gate(hands,box,size,goals,groups,goal_saturation=True).any()
    goals.zero_()
    assert group_contact_gate(hands,box,size,goals,groups,goal_saturation=True).all()
    assert not group_contact_gate(hands,box,size,goals,groups,goal_saturation=False).any()


def _common_observation(case):
    from test_before_object_alias import packet,observations
    if case.startswith('before_'):
        record=packet({'before_sit':0,'before_climb':1,'before_ontop':3}[case])
        groups=torch.tensor([[0.,0.,2.,3.]])
    elif case=='two_pairs':
        record=torch.tensor([[[1,2,0,4,12],[1,3,1,5,4],
                              [1,2,2,6,14],[1,3,3,7,6]]],dtype=torch.float)
        groups=torch.tensor([[0.,0.,2.,2.]])
    elif case=='coupled':
        groups=torch.zeros(1,4)
        record=task_graph_packet(compose_graph(torch.full((1,4),5),groups.long(),False),1).reshape(1,4,5)
    else:
        record=torch.tensor([[[1,0,0,0,4],[1,1,1,0,5],
                              [1,2,2,6,14],[1,4,3,7,15]]],dtype=torch.float)
        groups=torch.arange(4,dtype=torch.float)[None]
    # This helper creates equal object features and poses for alias duplicates.
    return torch.cat((observations(record),groups),-1)


@pytest.mark.parametrize('case',['before_sit','before_climb','before_ontop','two_pairs','coupled','solo'])
def test_common_policy_outputs_do_not_depend_on_environment_config(case):
    torch.set_num_threads(1);torch.manual_seed(731)
    before,coupled=[build(name) for name in NAMES]
    # Exercise learned coordination as well as the identity initialization.
    with torch.no_grad():
        for branch in (before.coordination,before.critic_coordination):
            for block in branch.blocks:
                block.attention.out_proj.weight.normal_(std=.01)
                block.ffn[-1].weight.normal_(std=.01)
    coupled.load_state_dict(before.state_dict(),strict=True)
    for net in (before,coupled):
        net.eval();net.relation_graph_spec['shuffle_token_order']=False
        net.coordination.shuffle_human_order=False;net.critic_coordination.shuffle_human_order=False
    obs=_common_observation(case);saved=obs.clone()
    with torch.no_grad():
        mu1,sigma1=before.eval_actor(obs);mu2,sigma2=coupled.eval_actor(obs)
        value1=before.eval_critic(obs);value2=coupled.eval_critic(obs)
    torch.testing.assert_close(mu1,mu2,rtol=0,atol=0)
    torch.testing.assert_close(sigma1,sigma2,rtol=0,atol=0)
    torch.testing.assert_close(value1,value2,rtol=0,atol=0)
    assert torch.equal(obs,saved), 'Alias rewriting must not mutate physical poses or the original task/group packet'


@pytest.mark.parametrize('case',['before_sit','before_climb','before_ontop','two_pairs','coupled','solo'])
def test_common_alias_keeps_reference_encoder_behavior_and_duplicate_tokens(case):
    from learning.multi_agent.before_network import BeforeBuilder
    from utils.before_object_alias import alias_layout,encoder_packet
    net=build(NAMES[0] if case.startswith('before') or case=='two_pairs' else NAMES[1])
    net.eval();net.relation_graph_spec['shuffle_token_order']=False
    obs=_common_observation(case);scene=obs[:,:-4];saved=scene.clone()
    record=scene[:,-20:];dependent,target,alias,_=alias_layout(record,4,8)
    poses=scene[:,1136:1248].reshape(-1,16,7)
    objects=scene[:,892:1132].reshape(-1,8,30)
    row,follower=dependent.nonzero(as_tuple=True)
    assert torch.equal(objects[row,alias[row,follower]-4],objects[row,target[row,follower]-4])
    assert torch.equal(poses[row,alias[row,follower]],poses[row,target[row,follower]])
    if case in ('coupled','solo'):
        assert not dependent.any() and torch.equal(encoder_packet(record,4,8),record)
    else:
        assert dependent.any()
        assert torch.equal(encoder_packet(record,4,8).reshape(-1,4,5)[row,follower,4],alias[row,follower].float())
    with torch.no_grad():
        for encoder in (net.actor_encoder,net.critic_encoder):
            expected=BeforeBuilder.Network._humans(net,encoder,scene)
            actual=net._humans(encoder,scene)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(scene,saved)


def test_coordination_biases_start_zero_and_checkpoint_loading_preserves_learned_bias():
    net=build(NAMES[0])
    for branch in (net.coordination,net.critic_coordination):
        for layer in branch.modules():
            if isinstance(layer,torch.nn.Linear) and layer.bias is not None:
                assert not layer.bias.any()
    state={key:value.clone() for key,value in net.state_dict().items()}
    key='coordination.blocks.0.ffn.0.bias'
    state[key].fill_(.123)
    net.load_state_dict(state,strict=True)
    torch.testing.assert_close(net.state_dict()[key],state[key],rtol=0,atol=0)
